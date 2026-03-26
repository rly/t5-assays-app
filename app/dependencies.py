from fastapi import Request, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth import SESSION_COOKIE_NAME, verify_session_token


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise _not_authenticated()
    user_id = verify_session_token(token)
    if user_id is None:
        raise _not_authenticated()
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise _not_authenticated()
    return user


class NotAuthenticated(Exception):
    pass


def _not_authenticated():
    return NotAuthenticated()
