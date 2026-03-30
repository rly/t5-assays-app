from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from app.database import init_db
from app.dependencies import NotAuthenticated
from app.routers import auth_routes, data_routes, chat_routes, api_routes, plot_routes, report_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _seed_default_users()
    yield


def _seed_default_users():
    """Create default users from DEFAULT_USERS env var on startup.
    Format: email1:password1,email2:password2
    """
    import os
    from app.database import SessionLocal
    from app.models import User, UserPreference
    from app.auth import hash_password

    users_str = os.environ.get("DEFAULT_USERS", "")
    if not users_str:
        return

    db = SessionLocal()
    try:
        for entry in users_str.split(","):
            entry = entry.strip()
            if ":" not in entry:
                continue
            email, password = entry.split(":", 1)
            existing = db.query(User).filter(User.email == email).first()
            if not existing:
                user = User(email=email, password_hash=hash_password(password))
                db.add(user)
                db.flush()
                db.add(UserPreference(user_id=user.id))
                db.commit()
    finally:
        db.close()


app = FastAPI(title="T5 Assays Data Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(auth_routes.router)
app.include_router(data_routes.router)
app.include_router(chat_routes.router)
app.include_router(api_routes.router)
app.include_router(plot_routes.router)
app.include_router(report_routes.router)


@app.exception_handler(NotAuthenticated)
async def not_authenticated_handler(request: Request, exc: NotAuthenticated):
    return RedirectResponse(url="/login", status_code=303)
