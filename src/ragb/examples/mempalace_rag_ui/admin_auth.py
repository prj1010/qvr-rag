"""Microsoft Entra ID (Microsoft SSO) authentication for the admin console."""

from __future__ import annotations

import os
import secrets
from typing import Any

from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import HTTPException, Request
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()


def _csv_env(name: str) -> set[str]:
    return {
        item.strip().lower()
        for item in os.getenv(name, "").split(",")
        if item.strip()
    }


class AdminAuth:
    """OIDC authorization-code flow with an explicit email allowlist."""

    def __init__(self) -> None:
        self.tenant_id = os.getenv("MICROSOFT_TENANT_ID", "").strip()
        self.client_id = os.getenv("MICROSOFT_CLIENT_ID", "").strip()
        self.client_secret = os.getenv("MICROSOFT_CLIENT_SECRET", "").strip()
        self.admin_emails = _csv_env("ADMIN_EMAILS")
        self.session_secret = os.getenv("SESSION_SECRET", "").strip()
        self.cookie_secure = os.getenv(
            "COOKIE_SECURE",
            "true" if os.getenv("RENDER") else "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.oauth = OAuth()
        if self.tenant_id and self.client_id and self.client_secret:
            self.oauth.register(
                name="microsoft",
                client_id=self.client_id,
                client_secret=self.client_secret,
                server_metadata_url=(
                    f"https://login.microsoftonline.com/{self.tenant_id}"
                    "/v2.0/.well-known/openid-configuration"
                ),
                client_kwargs={"scope": "openid profile email"},
            )

    @property
    def configured(self) -> bool:
        return bool(
            self.tenant_id
            and self.client_id
            and self.client_secret
            and self.admin_emails
            and self.session_secret
        )

    @property
    def session_signing_secret(self) -> str:
        # The admin flow refuses to start until SESSION_SECRET is explicitly set,
        # but a random fallback lets the public RAG workspace boot in development.
        return self.session_secret or secrets.token_urlsafe(32)

    def install_session_middleware(self, app: Any) -> None:
        app.add_middleware(
            SessionMiddleware,
            secret_key=self.session_signing_secret,
            session_cookie="mempalace_admin_session",
            max_age=8 * 60 * 60,
            same_site="lax",
            https_only=self.cookie_secure,
        )

    def ensure_configured(self) -> None:
        if not self.configured:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Microsoft SSO is not configured. Set MICROSOFT_TENANT_ID, "
                    "MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET, SESSION_SECRET, "
                    "and ADMIN_EMAILS."
                ),
            )

    async def login(self, request: Request) -> Any:
        self.ensure_configured()
        redirect_uri = request.url_for("admin_callback")
        return await self.oauth.microsoft.authorize_redirect(request, redirect_uri)

    async def callback(self, request: Request) -> dict[str, str]:
        self.ensure_configured()
        try:
            token = await self.oauth.microsoft.authorize_access_token(request)
        except OAuthError as exc:
            raise HTTPException(
                status_code=401,
                detail=f"Microsoft sign-in failed: {exc.error}",
            ) from exc
        user = token.get("userinfo") or {}
        email = self._email_from_user(user)
        if not email or email.lower() not in self.admin_emails:
            request.session.clear()
            raise HTTPException(
                status_code=403,
                detail="Your Microsoft account is authenticated but is not an admin.",
            )

        identity = {
            "email": email,
            "name": str(user.get("name") or email),
            "subject": str(user.get("sub") or ""),
        }
        request.session["admin"] = identity
        return identity

    @staticmethod
    def _email_from_user(user: dict[str, Any]) -> str:
        return str(
            user.get("preferred_username")
            or user.get("email")
            or user.get("upn")
            or ""
        ).strip()

    def current_admin(self, request: Request) -> dict[str, str] | None:
        identity = request.session.get("admin")
        if not isinstance(identity, dict):
            return None
        email = str(identity.get("email") or "").strip().lower()
        if not email or email not in self.admin_emails:
            request.session.clear()
            return None
        return {
            "email": email,
            "name": str(identity.get("name") or email),
            "subject": str(identity.get("subject") or ""),
        }

    def require_admin(self, request: Request) -> dict[str, str]:
        identity = self.current_admin(request)
        if identity is None:
            raise HTTPException(status_code=401, detail="Admin sign-in required.")
        return identity


ADMIN_AUTH = AdminAuth()
