from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings
from .constants import (
#    BEARER_TOKEN_TYPE,
    HTTP_UNAUTHORIZED,
)


security = HTTPBearer()


def create_access_token(username: str) -> str:
    """Create a JWT access token."""

    expiration = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_expiration_minutes
    )

    payload = {
        "sub": username,
        "exp": expiration,
    }

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Validate JWT and return the authenticated username."""

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )

        username = payload.get("sub")

        if not username:
            raise HTTPException(
                status_code=HTTP_UNAUTHORIZED,
                detail="Invalid authentication token",
            )

        return username

    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=HTTP_UNAUTHORIZED,
            detail="Authentication token has expired",
        ) from exc

    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=HTTP_UNAUTHORIZED,
            detail="Invalid authentication token",
        ) from exc