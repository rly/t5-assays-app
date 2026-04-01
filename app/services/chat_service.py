"""AI chat service: agent creation, tool calling, conversation management.

Uses Pydantic AI with OpenRouter to provide an agentic chat that can execute
Python code against the dataset via the run_python tool.
"""
import json
import time
from dataclasses import dataclass

import httpx
import pandas as pd
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from sqlalchemy.orm import Session

from app.config import settings, MODEL_MAPPING, FREE_MODELS
from app.models import Conversation, Message
from app.services.sandbox_service import execute_code

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
MAX_RECENT_MESSAGES = 6

CRITIC_SYSTEM_PROMPT = """You are a scientific reasoning critic reviewing an AI assistant's drug discovery data analysis response.

Your job: check the AI response against the actual tool outputs and flag problems.

Look for:
1. Numerical claims not supported by any tool output (hallucinations)
2. Internal contradictions (e.g., compound ranked differently in two places)
3. Conclusions that don't follow from the data shown
4. Important missing caveats (data quality, small sample size, assay limitations)

Respond ONLY with valid JSON in this exact format — no prose, no markdown fences:
{
  "verdict": "Pass" | "Minor issues" | "Significant issues",
  "issues": ["specific issue 1", "specific issue 2"],
  "caveats_missing": ["missing caveat 1"],
  "confidence_assessment": "One sentence summary of response reliability."
}

Rules:
- "Pass": response is well-supported and accurate
- "Minor issues": small inaccuracies or missing caveats, but overall sound
- "Significant issues": major hallucinations or unsupported claims
- Keep each issue to one sentence
- Return empty arrays [] if no issues or caveats are found"""

# Cache of model pricing: {model_id: {"prompt": float, "completion": float}}
_model_pricing: dict[str, dict[str, float]] = {}


async def _ensure_pricing_loaded():
    """Fetch model pricing from OpenRouter if not cached."""
    if _model_pricing:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(OPENROUTER_MODELS_URL)
            if resp.status_code == 200:
                for m in resp.json().get("data", []):
                    p = m.get("pricing", {})
                    try:
                        _model_pricing[m["id"]] = {
                            "prompt": float(p.get("prompt", 0)),
                            "completion": float(p.get("completion", 0)),
                        }
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass


def estimate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate cost from cached pricing. Returns None if pricing unavailable."""
    pricing = _model_pricing.get(model_id)
    if not pricing:
        return None
    return pricing["prompt"] * prompt_tokens + pricing["completion"] * completion_tokens


def get_api_key(user_key: str | None) -> str:
    """Return the user's key or the default key."""
    return user_key or settings.openrouter_default_api_key


def is_model_allowed(model_id: str, user_key: str | None) -> bool:
    """Check if the model can be used. Always allowed if a default key is configured."""
    if settings.openrouter_default_api_key:
        return True
    if model_id in FREE_MODELS:
        return True
    return bool(user_key)


def get_or_create_conversation(db: Session, user_id: int) -> Conversation:
    """Get or create the single conversation for a user."""
    conv = db.query(Conversation).filter(Conversation.user_id == user_id).first()
    if not conv:
        conv = Conversation(user_id=user_id)
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return conv


def save_message(db: Session, conversation_id: int, role: str, content: str, model_used: str = None, tokens: int = None, cost: float = None):
    msg = Message(
        conversation_id=conversation_id, role=role, content=content,
        model_used=model_used, tokens_used=tokens, cost=cost,
    )
    db.add(msg)
    db.commit()


def get_chat_messages(db: Session, conversation_id: int) -> list[dict]:
    msgs = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()
    return [{"role": m.role, "content": m.content, "tokens_used": m.tokens_used, "model_used": m.model_used} for m in msgs]


def cleanup_response(text: str) -> str:
    """Post-process AI response to fix common formatting issues."""
    lines = text.split("\n")
    result = []
    in_code_block = False
    unfenced_code_lines = []

    def flush_unfenced():
        if unfenced_code_lines:
            result.append("```python")
            result.extend(unfenced_code_lines)
            result.append("```")
            unfenced_code_lines.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_unfenced()
            in_code_block = not in_code_block
            result.append(line)
            continue
        if in_code_block:
            result.append(line)
            continue
        is_code_like = (
            stripped.startswith(("import ", "from ", "df[", "df.", "top", "print(", "for ", "if ", "result"))
            and not stripped.startswith(("import**", "importantly", "from the", "from a"))
            and ("=" in stripped or "(" in stripped or stripped.startswith("import "))
        )
        if is_code_like:
            unfenced_code_lines.append(line)
        else:
            flush_unfenced()
            result.append(line)

    flush_unfenced()
    return "\n".join(result)


@dataclass
class ChatDeps:
    """Dependencies passed to the agent tools at runtime."""
    datasets: dict[str, pd.DataFrame]  # {name: DataFrame}


def _describe_dataset(name: str, df: pd.DataFrame) -> str:
    """Generate column summary for a single dataset, with human-readable descriptions."""
    from app.column_descriptions import COLUMN_DESCRIPTIONS

    col_descriptions = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        human_desc = COLUMN_DESCRIPTIONS.get(col, "")
        label = f"{col} — {human_desc}" if human_desc else col
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() > non_null * 0.5 and numeric.notna().sum() > 0:
                desc = f"  - {label} (numeric, {non_null} non-null, min={numeric.min():.4g}, max={numeric.max():.4g}, mean={numeric.mean():.4g})"
            else:
                raise ValueError()
        except (ValueError, TypeError):
            n_unique = df[col].nunique()
            sample = df[col].dropna().head(3).tolist()
            desc = f"  - {label} (text, {non_null} non-null, {n_unique} unique, e.g. {sample})"
        col_descriptions.append(desc)

    var_name = name.replace(" ", "_").replace("-", "_")
    return (
        f'Dataset "{name}" (variable: {var_name}, {len(df)} rows x {len(df.columns)} cols):\n'
        + "\n".join(col_descriptions)
    )


def build_system_prompt(datasets: dict[str, pd.DataFrame]) -> str:
    """Build the system prompt with summaries for all provided datasets."""
    dataset_descriptions = []
    for name, df in datasets.items():
        dataset_descriptions.append(_describe_dataset(name, df))

    data_context = "\n\n".join(dataset_descriptions)

    # Build variable list for the prompt
    var_list = []
    for name in datasets:
        var_name = name.replace(" ", "_").replace("-", "_")
        var_list.append(f'  - {var_name} (or datasets["{name}"])')

    if len(datasets) == 1:
        access_note = "For convenience, the single dataset is also available as `df`.\n"
    else:
        access_note = ""

    return (
        "You are a helpful data analyst assistant for alphaviral macrodomain assay data. "
        "You help scientists identify promising small molecule therapeutic candidates.\n\n"
        "IMPORTANT FORMATTING RULES:\n"
        "- Always format your responses in Markdown.\n"
        "- Use bullet points, numbered lists, and bold text for readability.\n"
        "- Keep explanations concise and focused on actionable insights.\n"
        "- When showing data or results, use fenced code blocks: ```\\n...\\n```\n\n"
        "COMPUTATION:\n"
        "- You have a `run_python` tool that executes Python code.\n"
        f"- Available datasets ({len(datasets)}):\n" + "\n".join(var_list) + "\n"
        f"{access_note}"
        "- ALWAYS use the run_python tool to access, query, or analyze data. You can see column summaries below but NOT the raw data.\n"
        "- Use print() in your code to produce output.\n"
        "- You can call the tool multiple times to do multi-step analysis.\n"
        "- NEVER guess, fabricate, or estimate data values. Always compute them using run_python.\n"
        "- If your code produces an error, fix it and try again.\n\n"
        "CHEMINFORMATICS (RDKit):\n"
        "RDKit is available for molecular analysis. The 'Structure' column (when present) contains SMILES strings.\n"
        "Quick reference:\n"
        "  from rdkit import Chem\n"
        "  from rdkit.Chem import AllChem, Descriptors, Draw\n"
        "  from rdkit import DataStructs\n\n"
        "  # Parse SMILES\n"
        "  mol = Chem.MolFromSmiles(smiles_string)  # returns None if invalid\n\n"
        "  # Molecular descriptors\n"
        "  Descriptors.MolWt(mol)           # molecular weight\n"
        "  Descriptors.MolLogP(mol)         # LogP\n"
        "  Descriptors.TPSA(mol)            # topological polar surface area\n"
        "  Descriptors.NumHDonors(mol)      # H-bond donors\n"
        "  Descriptors.NumHAcceptors(mol)   # H-bond acceptors\n"
        "  Descriptors.NumRotatableBonds(mol)\n"
        "  Descriptors.CalcMolDescriptors(mol)  # dict of all descriptors\n\n"
        "  # Lipinski Rule of Five: MW<=500, LogP<=5, HBD<=5, HBA<=10\n\n"
        "  # Fingerprints & similarity\n"
        "  fpgen = AllChem.GetMorganGenerator(radius=2)\n"
        "  fp = fpgen.GetFingerprint(mol)\n"
        "  DataStructs.TanimotoSimilarity(fp1, fp2)  # 0-1 similarity\n\n"
        "  # Substructure search\n"
        "  pattern = Chem.MolFromSmarts('[OH]')  # SMARTS pattern\n"
        "  mol.HasSubstructMatch(pattern)  # True/False\n"
        "  mol.GetSubstructMatches(pattern)  # atom indices\n\n"
        "  # Bulk: compute descriptors for a DataFrame column of SMILES\n"
        "  df['mol'] = df['Structure'].apply(Chem.MolFromSmiles)\n"
        "  df['MW'] = df['mol'].apply(lambda m: Descriptors.MolWt(m) if m else None)\n\n"
        "  # Plotting — ALWAYS use print(fig.to_json()) to output figures:\n"
        "  import plotly.express as px\n"
        "  fig = px.scatter(df, x='LogP', y='TPSA', hover_data=['Name'], title='LogP vs TPSA')\n"
        "  print(fig.to_json())  # fig.show() does nothing in the sandbox\n\n"
        "Use RDKit when asked about molecular properties, structural similarity, substructure searches,\n"
        "drug-likeness filtering, or SAR (structure-activity relationship) analysis.\n\n"
        "CHEMINFORMATICS TOOLS (faster than writing RDKit code by hand):\n"
        "1. `compute_descriptors(smiles_list, names=None)`\n"
        "   Compute MW, LogP, TPSA, HBD, HBA, RotBonds, QED, and Lipinski pass/fail for a list of SMILES.\n"
        "   Returns a JSON table. Use this instead of writing descriptor loops in run_python.\n\n"
        "2. `cluster_by_scaffold(smiles_list, names=None, cutoff=0.4)`\n"
        "   Murcko scaffold decomposition + Butina fingerprint clustering.\n"
        "   Returns cluster assignments and scaffold SMILES for each compound.\n"
        "   Use to answer 'how many distinct chemical series are in the top hits?'\n\n"
        "3. `compute_tanimoto_matrix(smiles_list, names=None)`\n"
        "   Pairwise Morgan fingerprint Tanimoto similarity matrix for up to 100 compounds.\n"
        "   Returns a JSON matrix. Use for diversity analysis or picking a representative set.\n\n"
        "4. `predict_admet(smiles_list, names=None)`\n"
        "   Rule-based ADMET prediction using RDKit (no external API required).\n"
        "   Returns GI absorption, BBB permeability, P-gp substrate likelihood, CYP inhibition\n"
        "   alerts (1A2, 2C9, 2C19, 2D6, 3A4), PAINS flags, and Brenk structural alerts.\n"
        "   Use to flag problematic compounds before prioritizing hits.\n\n"
        "PUBCHEM TOOLS:\n"
        "You have four tools for querying the PubChem database. All make live network requests.\n\n"
        "1. `lookup_pubchem(identifiers, identifier_type='smiles')`\n"
        "   Look up compounds by SMILES, name, or CID. Returns: cid, iupac_name, synonyms,\n"
        "   molecular_formula, molecular_weight, xlogp, tpsa, h_bond_donor/acceptor_count,\n"
        "   rotatable_bond_count, isomeric_smiles.\n"
        "   Use for: compound identity, trade names, drug approval status, cross-referencing.\n\n"
        "2. `search_pubchem_by_substructure(smarts, max_results=20)`\n"
        "   Find all PubChem compounds containing a given SMARTS substructure.\n"
        "   Use for: scaffold enumeration, finding all known compounds with a specific motif.\n\n"
        "3. `search_pubchem_by_similarity(smiles, threshold=90, max_results=20)`\n"
        "   Find PubChem compounds with Tanimoto similarity >= threshold (0-100) to a query SMILES.\n"
        "   Use for: analog discovery, scaffold hopping, finding commercially available analogs.\n\n"
        "4. `get_pubchem_bioassays(cid)`\n"
        "   Retrieve bioassay activity summary for a PubChem CID — which assays it was tested in\n"
        "   and whether it was active or inactive.\n"
        "   Use for: prior art, PAINS/frequent hitter detection, target selectivity profiling.\n\n"
        "Workflow: use run_python to extract SMILES/names from the dataset, then call PubChem tools.\n"
        "Do NOT use pubchempy or make network requests inside run_python.\n\n"
        f"Data context:\n{data_context}\n\n"
        "Answer questions accurately. Highlight key findings and actionable insights."
    )


def create_agent(api_key: str, model_id: str) -> Agent[ChatDeps, str]:
    """Create a Pydantic AI agent configured for OpenRouter."""
    model = OpenRouterModel(
        model_id,
        provider=OpenRouterProvider(api_key=api_key),
    )

    agent = Agent(
        model,
        deps_type=ChatDeps,
        output_type=str,
    )

    @agent.tool
    async def lookup_pubchem(ctx: RunContext[ChatDeps], identifiers: list[str], identifier_type: str = "smiles") -> str:
        """Look up compound data from PubChem for a list of SMILES strings, names, or CIDs.

        Args:
            identifiers: List of SMILES strings, compound names, or CID numbers (as strings).
            identifier_type: One of 'smiles', 'name', or 'cid'. Defaults to 'smiles'.

        Returns:
            JSON string with a list of compound records containing cid, iupac_name, synonyms,
            molecular_formula, molecular_weight, xlogp, tpsa, h_bond_donor_count,
            h_bond_acceptor_count, rotatable_bond_count, and isomeric_smiles.
        """
        import pubchempy as pcp

        results = []
        for ident in identifiers[:20]:  # cap at 20 to avoid excessive API calls
            try:
                compounds = pcp.get_compounds(ident, identifier_type)
                if compounds:
                    c = compounds[0]
                    results.append({
                        "query": ident,
                        "cid": c.cid,
                        "iupac_name": c.iupac_name,
                        "synonyms": (c.synonyms or [])[:5],
                        "molecular_formula": c.molecular_formula,
                        "molecular_weight": c.molecular_weight,
                        "xlogp": c.xlogp,
                        "tpsa": c.tpsa,
                        "h_bond_donor_count": c.h_bond_donor_count,
                        "h_bond_acceptor_count": c.h_bond_acceptor_count,
                        "rotatable_bond_count": c.rotatable_bond_count,
                        "isomeric_smiles": c.isomeric_smiles,
                    })
                else:
                    results.append({"query": ident, "error": "Not found in PubChem"})
            except Exception as e:
                results.append({"query": ident, "error": str(e)})

        return json.dumps(results, indent=2)

    @agent.tool
    async def search_pubchem_by_substructure(ctx: RunContext[ChatDeps], smarts: str, max_results: int = 20) -> str:
        """Search PubChem for compounds containing a given substructure (SMARTS pattern).

        Args:
            smarts: SMARTS pattern describing the substructure to search for (e.g. 'c1ccccc1' for benzene).
            max_results: Maximum number of results to return (default 20, max 100).

        Returns:
            JSON string with a list of matching compound records from PubChem.
        """
        import pubchempy as pcp

        max_results = min(max_results, 100)
        try:
            compounds = pcp.get_compounds(smarts, "smarts", listkey_count=max_results)
            results = []
            for c in compounds[:max_results]:
                results.append({
                    "cid": c.cid,
                    "iupac_name": c.iupac_name,
                    "synonyms": (c.synonyms or [])[:3],
                    "molecular_formula": c.molecular_formula,
                    "molecular_weight": c.molecular_weight,
                    "isomeric_smiles": c.isomeric_smiles,
                    "xlogp": c.xlogp,
                    "tpsa": c.tpsa,
                })
            return json.dumps({"query_smarts": smarts, "count": len(results), "compounds": results}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @agent.tool
    async def search_pubchem_by_similarity(ctx: RunContext[ChatDeps], smiles: str, threshold: int = 90, max_results: int = 20) -> str:
        """Search PubChem for compounds structurally similar to a given SMILES string.

        Args:
            smiles: SMILES string of the query compound.
            threshold: Tanimoto similarity threshold (0-100, default 90 = 90% similar).
            max_results: Maximum number of results to return (default 20, max 100).

        Returns:
            JSON string with a list of similar compound records from PubChem.
        """
        import pubchempy as pcp

        max_results = min(max_results, 100)
        threshold = max(0, min(threshold, 100))
        try:
            compounds = pcp.get_compounds(
                smiles, "smiles",
                searchtype="similarity",
                Threshold=threshold,
                listkey_count=max_results,
            )
            results = []
            for c in compounds[:max_results]:
                results.append({
                    "cid": c.cid,
                    "iupac_name": c.iupac_name,
                    "synonyms": (c.synonyms or [])[:3],
                    "molecular_formula": c.molecular_formula,
                    "molecular_weight": c.molecular_weight,
                    "isomeric_smiles": c.isomeric_smiles,
                    "xlogp": c.xlogp,
                    "tpsa": c.tpsa,
                })
            return json.dumps({"query_smiles": smiles, "threshold": threshold, "count": len(results), "compounds": results}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @agent.tool
    async def get_pubchem_bioassays(ctx: RunContext[ChatDeps], cid: int) -> str:
        """Retrieve bioassay activity summary for a PubChem compound CID.

        Returns which assays the compound was tested in and whether it was active or inactive.
        Useful for determining if a compound has been tested against viral or other biological targets.

        Args:
            cid: PubChem Compound ID (integer).

        Returns:
            JSON string with bioassay activity summary records.
        """
        import httpx

        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/assaysummary/JSON"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
            if resp.status_code == 404:
                return json.dumps({"cid": cid, "error": "No bioassay data found for this CID"})
            resp.raise_for_status()
            data = resp.json()
            table = data.get("Table", {})
            columns = [col["Name"] for col in table.get("Columns", {}).get("Column", [])]
            rows = table.get("Row", [])
            records = []
            for row in rows[:50]:  # cap at 50 assays
                cells = row.get("Cell", [])
                records.append(dict(zip(columns, cells)))
            return json.dumps({"cid": cid, "assay_count": len(rows), "assays": records}, indent=2)
        except Exception as e:
            return json.dumps({"cid": cid, "error": str(e)})

    @agent.tool
    def compute_descriptors(
        ctx: RunContext[ChatDeps],
        smiles_list: list[str],
        names: list[str] | None = None,
    ) -> str:
        """Compute molecular descriptors for a list of SMILES strings.

        Calculates MW, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, QED,
        and Lipinski Rule-of-Five pass/fail for each compound.

        Args:
            smiles_list: List of SMILES strings to process (max 500).
            names: Optional list of compound names, same length as smiles_list.

        Returns:
            JSON string with a list of descriptor records, one per compound.
            Invalid SMILES are included with an 'error' field instead of descriptors.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED
        except ImportError:
            return json.dumps({"error": "RDKit is not installed"})

        smiles_list = smiles_list[:500]
        if names and len(names) != len(smiles_list):
            names = None

        records = []
        for i, smi in enumerate(smiles_list):
            name = names[i] if names else None
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rec = {"smiles": smi, "error": "Invalid SMILES"}
                if name:
                    rec["name"] = name
                records.append(rec)
                continue
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotb = Descriptors.NumRotatableBonds(mol)
            qed = round(QED.qed(mol), 4)
            lipinski = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
            rec = {
                "smiles": smi,
                "MW": round(mw, 3),
                "LogP": round(logp, 3),
                "TPSA": round(tpsa, 3),
                "HBD": hbd,
                "HBA": hba,
                "RotBonds": rotb,
                "QED": qed,
                "Lipinski_pass": lipinski,
            }
            if name:
                rec["name"] = name
            records.append(rec)

        return json.dumps({"count": len(records), "descriptors": records}, indent=2)

    @agent.tool
    def cluster_by_scaffold(
        ctx: RunContext[ChatDeps],
        smiles_list: list[str],
        names: list[str] | None = None,
        cutoff: float = 0.4,
    ) -> str:
        """Cluster compounds by Murcko scaffold and Morgan fingerprint similarity.

        First extracts the Murcko scaffold for each compound, then performs
        Butina clustering on Morgan fingerprints (radius=2) using the given
        distance cutoff (1 - Tanimoto similarity).

        Args:
            smiles_list: List of SMILES strings (max 500).
            names: Optional list of compound names, same length as smiles_list.
            cutoff: Butina distance cutoff (default 0.4 = compounds with Tanimoto >= 0.6
                    end up in the same cluster). Lower values = tighter clusters.

        Returns:
            JSON string with cluster assignments, scaffold SMILES, and cluster sizes.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            from rdkit import DataStructs
            from rdkit.ML.Cluster import Butina
        except ImportError:
            return json.dumps({"error": "RDKit is not installed"})

        smiles_list = smiles_list[:500]
        if names and len(names) != len(smiles_list):
            names = None

        fpgen = AllChem.GetMorganGenerator(radius=2)
        mols, fps, valid_idx = [], [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mols.append(mol)
                fps.append(fpgen.GetFingerprint(mol))
                valid_idx.append(i)

        if not fps:
            return json.dumps({"error": "No valid SMILES provided"})

        # Butina clustering
        n = len(fps)
        dists = []
        for i in range(1, n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - s for s in sims])
        clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

        # Build scaffold map
        scaffold_map = {}
        for mol in mols:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_map[Chem.MolToSmiles(mol)] = Chem.MolToSmiles(scaffold)
            except Exception:
                scaffold_map[Chem.MolToSmiles(mol)] = ""

        # Assign cluster IDs
        compound_clusters = [None] * n
        for cluster_id, member_indices in enumerate(clusters):
            for idx in member_indices:
                compound_clusters[idx] = cluster_id

        records = []
        for local_i, orig_i in enumerate(valid_idx):
            smi = smiles_list[orig_i]
            mol_smi = Chem.MolToSmiles(mols[local_i])
            rec = {
                "smiles": smi,
                "cluster_id": compound_clusters[local_i],
                "scaffold_smiles": scaffold_map.get(mol_smi, ""),
            }
            if names:
                rec["name"] = names[orig_i]
            records.append(rec)

        cluster_sizes = {}
        for r in records:
            cid = r["cluster_id"]
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

        return json.dumps({
            "num_clusters": len(clusters),
            "num_compounds": len(records),
            "cutoff": cutoff,
            "cluster_sizes": {str(k): v for k, v in sorted(cluster_sizes.items())},
            "compounds": records,
        }, indent=2)

    @agent.tool
    def compute_tanimoto_matrix(
        ctx: RunContext[ChatDeps],
        smiles_list: list[str],
        names: list[str] | None = None,
    ) -> str:
        """Compute a pairwise Tanimoto similarity matrix for a list of SMILES.

        Uses Morgan fingerprints (radius=2). Returns a symmetric matrix where
        entry [i][j] is the Tanimoto similarity between compound i and compound j.

        Args:
            smiles_list: List of SMILES strings (max 100).
            names: Optional list of compound names for labelling rows/columns.

        Returns:
            JSON string with labels and the similarity matrix (list of lists).
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit import DataStructs
        except ImportError:
            return json.dumps({"error": "RDKit is not installed"})

        smiles_list = smiles_list[:100]
        if names and len(names) != len(smiles_list):
            names = None

        fpgen = AllChem.GetMorganGenerator(radius=2)
        fps, labels, valid_smiles = [], [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append(fpgen.GetFingerprint(mol))
                labels.append(names[i] if names else smi)
                valid_smiles.append(smi)

        if not fps:
            return json.dumps({"error": "No valid SMILES provided"})

        n = len(fps)
        matrix = []
        for i in range(n):
            row = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            matrix.append([round(v, 4) for v in row])

        return json.dumps({
            "num_compounds": n,
            "labels": labels,
            "matrix": matrix,
        }, indent=2)

    @agent.tool
    def predict_admet(
        ctx: RunContext[ChatDeps],
        smiles_list: list[str],
        names: list[str] | None = None,
    ) -> str:
        """Predict ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties.

        Uses rule-based heuristics and RDKit's built-in PAINS/Brenk filter catalogs.
        No external API required. Suitable for rapid triage of hit lists.

        Properties predicted:
        - GI absorption: High if RotBonds <= 10 AND TPSA <= 140 (Veber rules)
        - BBB permeability: Likely if MW < 450, TPSA < 90, LogP in [0,5], HBD <= 3
        - P-gp substrate: Likely if MW > 400 OR TPSA > 75
        - CYP1A2 inhibitor: Planar aromatic amines / furans (SMARTS-based)
        - CYP2C9 inhibitor: Acidic group + aromatic ring
        - CYP2C19 inhibitor: Imidazole or pyridine motifs
        - CYP2D6 inhibitor: Basic nitrogen within 2 bonds of aromatic ring
        - CYP3A4 inhibitor: MW > 400 with >= 3 aromatic rings
        - PAINS alerts: RDKit FilterCatalog (PAINS_A/B/C)
        - Brenk alerts: RDKit FilterCatalog (Brenk structural alerts)
        - Drug-likeness: Lipinski Ro5 pass/fail

        Args:
            smiles_list: List of SMILES strings (max 500).
            names: Optional list of compound names, same length as smiles_list.

        Returns:
            JSON string with ADMET predictions for each compound.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        except ImportError:
            return json.dumps({"error": "RDKit is not installed"})

        smiles_list = smiles_list[:500]
        if names and len(names) != len(smiles_list):
            names = None

        # Build PAINS and Brenk filter catalogs
        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        pains_catalog = FilterCatalog(pains_params)

        brenk_params = FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        brenk_catalog = FilterCatalog(brenk_params)

        # CYP inhibition SMARTS (simplified heuristics, not ML-based)
        _CYP_SMARTS = {
            "CYP1A2": [
                "[nH]1cccc1",           # pyrrole
                "c1ccc2[nH]ccc2c1",     # indole
                "c1ccncc1N",            # aminopyridine
                "[NH2]c1ccccc1",        # aniline
                "o1cccc1",              # furan
            ],
            "CYP2C9": [
                "[OH,O-]C(=O)",         # carboxylic acid / carboxylate
                "S(=O)(=O)[OH,O-]",     # sulfonic acid
                "c1ccccc1C(=O)[OH]",    # benzoic acid motif
            ],
            "CYP2C19": [
                "c1cnc[nH]1",           # imidazole
                "c1ccncc1",             # pyridine
                "c1ncc[nH]1",           # imidazole variant
            ],
            "CYP2D6": [
                "[NH,NH2,NH3+,n]~[CH2,CH]~[CH2,CH]~c1ccccc1",  # basic N near aromatic
                "[NH,NH2]Cc1ccccc1",    # benzylamine
            ],
            "CYP3A4": [
                # Handled by MW + ring count heuristic below
            ],
        }

        compiled_cyp = {}
        for cyp, smarts_list in _CYP_SMARTS.items():
            compiled_cyp[cyp] = [Chem.MolFromSmarts(s) for s in smarts_list if Chem.MolFromSmarts(s)]

        records = []
        for i, smi in enumerate(smiles_list):
            name = names[i] if names else None
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rec = {"smiles": smi, "error": "Invalid SMILES"}
                if name:
                    rec["name"] = name
                records.append(rec)
                continue

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotb = Descriptors.NumRotatableBonds(mol)
            n_aromatic_rings = Descriptors.NumAromaticRings(mol)

            # Absorption
            gi_absorption = "High" if rotb <= 10 and tpsa <= 140 else "Low"

            # Distribution
            bbb = (mw < 450 and tpsa < 90 and 0 <= logp <= 5 and hbd <= 3)

            # P-gp substrate (efflux pump — reduces CNS and oral bioavailability)
            pgp_substrate = mw > 400 or tpsa > 75

            # CYP inhibition
            cyp_flags = {}
            for cyp, patterns in compiled_cyp.items():
                cyp_flags[cyp] = any(mol.HasSubstructMatch(p) for p in patterns)
            # CYP3A4: large MW + multiple aromatic rings
            cyp_flags["CYP3A4"] = mw > 400 and n_aromatic_rings >= 3

            # PAINS
            pains_matches = pains_catalog.GetMatches(mol)
            pains_alerts = [m.GetDescription() for m in pains_matches]

            # Brenk
            brenk_matches = brenk_catalog.GetMatches(mol)
            brenk_alerts = [m.GetDescription() for m in brenk_matches]

            # Lipinski
            lipinski = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10

            rec = {
                "smiles": smi,
                "GI_absorption": gi_absorption,
                "BBB_permeable": bbb,
                "Pgp_substrate": pgp_substrate,
                "CYP1A2_inhibitor": cyp_flags["CYP1A2"],
                "CYP2C9_inhibitor": cyp_flags["CYP2C9"],
                "CYP2C19_inhibitor": cyp_flags["CYP2C19"],
                "CYP2D6_inhibitor": cyp_flags["CYP2D6"],
                "CYP3A4_inhibitor": cyp_flags["CYP3A4"],
                "PAINS_alerts": pains_alerts,
                "Brenk_alerts": brenk_alerts,
                "Lipinski_pass": lipinski,
                "num_PAINS": len(pains_alerts),
                "num_Brenk": len(brenk_alerts),
            }
            if name:
                rec["name"] = name
            records.append(rec)

        # Summary counts
        valid = [r for r in records if "error" not in r]
        summary = {
            "total": len(records),
            "valid": len(valid),
            "high_GI_absorption": sum(1 for r in valid if r["GI_absorption"] == "High"),
            "BBB_permeable": sum(1 for r in valid if r["BBB_permeable"]),
            "Pgp_substrate": sum(1 for r in valid if r["Pgp_substrate"]),
            "Lipinski_pass": sum(1 for r in valid if r["Lipinski_pass"]),
            "any_PAINS": sum(1 for r in valid if r["num_PAINS"] > 0),
            "any_Brenk": sum(1 for r in valid if r["num_Brenk"] > 0),
        }

        return json.dumps({"summary": summary, "compounds": records}, indent=2)

    @agent.tool
    def run_python(ctx: RunContext[ChatDeps], code: str) -> str:
        """Execute Python code against the provided datasets.
        Available variables: each dataset as a named variable (e.g., VEEV_PEITHO_SPR), plus a 'datasets' dict.
        If only one dataset is provided, 'df' is also available.
        Use print() to output results. pandas, numpy, rdkit, math, statistics, re, datetime are available.
        Do NOT use pubchempy or make network requests in run_python — use the lookup_pubchem tool instead."""
        result = execute_code(code, ctx.deps.datasets)
        if result["success"]:
            output = result.get("output", "").strip()
            return output if output else "(no output — make sure to use print())"
        else:
            error = result.get("error", "Unknown error")
            error_lines = error.strip().split("\n")
            return f"Error: {error_lines[-1]}"

    return agent


async def run_critic(
    api_key: str,
    model_id: str,
    user_message: str,
    tool_steps: list[dict],
    primary_response: str,
) -> dict | None:
    """Run a lightweight critic agent to review the primary agent's response.

    Checks for hallucinations, unsupported claims, contradictions, and missing caveats.
    Only runs when the primary agent made at least one tool call (so there is ground-truth
    computed output to verify against).

    Returns a dict with keys: verdict, issues, caveats_missing, confidence_assessment.
    Returns None if the critic fails or there are no tool calls to verify.
    """
    if not tool_steps:
        return None

    # Build a compact summary of tool calls + outputs (cap length to control tokens)
    tool_parts: list[str] = []
    i = 0
    step_num = 0
    while i < len(tool_steps):
        step = tool_steps[i]
        if step["type"] == "call":
            step_num += 1
            tool_name = step.get("tool", "unknown")
            output = ""
            if i + 1 < len(tool_steps) and tool_steps[i + 1]["type"] == "return":
                output = tool_steps[i + 1].get("output", "")
                i += 1
            tool_parts.append(f"[Tool {step_num}: {tool_name}]\n{output}")
        i += 1

    tool_summary = "\n\n".join(tool_parts)
    critic_prompt = (
        f"USER QUESTION:\n{user_message[:500]}\n\n"
        f"TOOL OUTPUTS:\n{tool_summary}\n\n"
        f"AI RESPONSE:\n{primary_response[:3000]}\n\n"
        "Review and respond with JSON only."
    )

    try:
        model = OpenRouterModel(model_id, provider=OpenRouterProvider(api_key=api_key))
        critic_agent = Agent(model, output_type=str)
        result = await critic_agent.run(critic_prompt, instructions=CRITIC_SYSTEM_PROMPT)
        raw = result.output.strip()
        # Strip markdown code fences if the model wraps the JSON
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()
        return json.loads(raw)
    except Exception:
        return None


async def run_agent_chat(
    api_key: str, model_id: str, system_prompt: str,
    conversation_messages: list[dict], user_message: str,
    datasets: dict[str, pd.DataFrame],
    conversation_summary: str | None = None,
) -> dict:
    """Run the agent with tool calling. Returns {"content": str, "usage": dict, "tool_steps": list, "elapsed": float}."""
    await _ensure_pricing_loaded()

    agent = create_agent(api_key, model_id)

    # Build message history for the agent
    # Pydantic AI expects its own message format, but we store simple role/content pairs.
    # We'll pass conversation context via the system prompt instead of message_history,
    # since our stored messages don't include tool call details.
    history_text = ""
    previous = conversation_messages[:]
    if len(previous) > MAX_RECENT_MESSAGES:
        recent = previous[-MAX_RECENT_MESSAGES:]
        if conversation_summary:
            history_text = f"[Previous conversation summary: {conversation_summary}]\n\n"
        for m in recent:
            role = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n\n"
    else:
        for m in previous:
            role = "User" if m["role"] == "user" else "Assistant"
            history_text += f"{role}: {m['content']}\n\n"

    # Combine system prompt with history context
    full_system_prompt = system_prompt
    if history_text:
        full_system_prompt += f"\n\nConversation history:\n{history_text}"

    # Run the agent with timing
    start_time = time.monotonic()

    result = await agent.run(
        user_message,
        deps=ChatDeps(datasets=datasets),
        instructions=full_system_prompt,
    )

    elapsed = time.monotonic() - start_time

    content = result.output
    usage = result.usage()

    usage_info = {
        "prompt_tokens": usage.input_tokens or 0,
        "completion_tokens": usage.output_tokens or 0,
        "total_tokens": (usage.input_tokens or 0) + (usage.output_tokens or 0),
    }

    # Estimate cost
    pt = usage_info["prompt_tokens"]
    ct = usage_info["completion_tokens"]
    if pt or ct:
        est = estimate_cost(model_id, pt, ct)
        if est is not None:
            usage_info["cost"] = est
            usage_info["cost_estimated"] = True

    # Extract tool call steps from message history
    tool_steps = []
    for msg in result.all_messages():
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if part.part_kind == "tool-call":
                    tool_steps.append({
                        "type": "call",
                        "tool": part.tool_name,
                        "args": part.args if isinstance(part.args, str) else json.dumps(part.args),
                    })
                elif part.part_kind == "tool-return":
                    tool_steps.append({
                        "type": "return",
                        "tool": part.tool_name,
                        "output": str(part.content),
                    })

    # Run critic when the primary agent made tool calls (ground-truth outputs exist to verify)
    critique: dict | None = None
    if tool_steps:
        critique = await run_critic(api_key, model_id, user_message, tool_steps, content)

    return {"content": content, "usage": usage_info, "tool_steps": tool_steps, "elapsed": elapsed, "critique": critique}




def generate_summary_prompt(dataset_names: list[str]) -> str:
    """Generate a summary prompt based on the provided dataset names."""
    if not dataset_names:
        return "No datasets are provided. Please select datasets to provide to the AI first."
    names = ", ".join(dataset_names)
    return f"Summarize the provided datasets ({names}). Highlight key findings, patterns, and contrasts between different assay results."


async def summarize_conversation(messages: list[dict], api_key: str, model_id: str) -> str | None:
    """Summarize a list of messages into a brief summary."""
    if not messages:
        return None

    conversation_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    prompt = (
        "Summarize the following conversation concisely, preserving key information, "
        "questions asked, and conclusions reached:\n\n" + conversation_text
    )

    try:
        model = OpenRouterModel(
            model_id,
            provider=OpenRouterProvider(api_key=api_key),
        )
        agent = Agent(model)
        result = await agent.run(prompt)
        return result.output
    except Exception:
        return None
