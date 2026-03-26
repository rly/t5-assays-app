from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from app.database import init_db
from app.dependencies import NotAuthenticated
from app.routers import auth_routes, data_routes, chat_routes, api_routes, plot_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="T5 Assays Data Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(auth_routes.router)
app.include_router(data_routes.router)
app.include_router(chat_routes.router)
app.include_router(api_routes.router)
app.include_router(plot_routes.router)


@app.exception_handler(NotAuthenticated)
async def not_authenticated_handler(request: Request, exc: NotAuthenticated):
    return RedirectResponse(url="/login", status_code=303)
