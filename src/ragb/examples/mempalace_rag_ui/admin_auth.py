"""Microsoft Entra ID (Microsoft SSO) authentication for the admin console."""

from __future__ import annotations

import os
import secrets
from html import escape
from typing import Any

from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
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
        self.fallback_enabled = os.getenv(
            "ADMIN_FALLBACK_ENABLED", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.fallback_token = os.getenv("ADMIN_FALLBACK_TOKEN", "").strip()
        self.fallback_email = (
            os.getenv("ADMIN_FALLBACK_EMAIL", "fallback-admin@local").strip()
            or "fallback-admin@local"
        )
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
    def fallback_configured(self) -> bool:
        """Whether the explicitly enabled temporary token fallback is usable."""

        return (
            self.fallback_enabled
            and len(self.fallback_token) >= 16
            and bool(self.session_secret)
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

    def ensure_fallback_configured(self) -> None:
        if not self.fallback_configured:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Admin fallback is disabled. Set ADMIN_FALLBACK_ENABLED=true "
                    "and ADMIN_FALLBACK_TOKEN to a random token of at least 16 "
                    "characters; SESSION_SECRET is also required."
                ),
            )

    async def login(self, request: Request) -> Any:
        if not self.configured:
            if self.fallback_configured:
                from starlette.responses import RedirectResponse

                return RedirectResponse(url="/admin/fallback", status_code=303)
            self.ensure_configured()
        redirect_uri = request.url_for("admin_callback")
        return await self.oauth.microsoft.authorize_redirect(request, redirect_uri)

    def fallback_page(self) -> HTMLResponse:
        self.ensure_fallback_configured()
        email = escape(self.fallback_email)
        return HTMLResponse(
            f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Admin fallback sign-in</title><style>
:root{{color-scheme:dark}}body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#080b18;color:#eef0f8;font:15px system-ui,sans-serif}}main{{width:min(390px,calc(100% - 40px));padding:28px;border:1px solid #303756;border-radius:16px;background:#11172b;box-shadow:0 20px 70px #0008}}h1{{margin:0 0 8px;font-size:22px}}p{{color:#aeb7d1;font-size:13px;line-height:1.55}}label{{display:block;margin:20px 0 7px;color:#bac2d9;font-size:12px}}input{{box-sizing:border-box;width:100%;height:42px;padding:0 11px;border:1px solid #3d476b;border-radius:8px;background:#080b18;color:#fff;font-size:14px}}button{{width:100%;height:42px;margin-top:16px;border:0;border-radius:8px;background:#765ee5;color:#fff;font-weight:700;cursor:pointer}}small{{display:block;margin-top:16px;color:#7783a2;font-size:11px}}code{{color:#c9bdff}}</style></head>
<body><main><h1>Temporary admin access</h1><p>Microsoft SSO is not configured. Enter the fallback token to open the protected observability and governance console.</p><form method="post" action="/admin/fallback"><label for="token">Fallback token</label><input id="token" name="token" type="password" autocomplete="current-password" required autofocus><button type="submit">Sign in</button></form><small>Fallback identity: <code>{email}</code>. Disable this fallback after Microsoft SSO is ready.</small></main></body></html>""",
            headers={"Cache-Control": "no-store"},
        )

    def create_fallback_session(self, request: Request, token: str) -> dict[str, str]:
        self.ensure_fallback_configured()
        if not secrets.compare_digest(token.strip(), self.fallback_token):
            raise HTTPException(status_code=401, detail="Invalid fallback token.")
        identity = {
            "email": self.fallback_email.lower(),
            "name": "Fallback admin",
            "subject": "fallback",
            "auth_method": "fallback",
        }
        request.session["admin"] = identity
        return identity

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
            "auth_method": "microsoft",
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
        if identity.get("auth_method") == "fallback":
            if not self.fallback_configured:
                request.session.clear()
                return None
            return {
                "email": str(identity.get("email") or self.fallback_email),
                "name": str(identity.get("name") or "Fallback admin"),
                "subject": "fallback",
                "auth_method": "fallback",
            }
        email = str(identity.get("email") or "").strip().lower()
        if not email or email not in self.admin_emails:
            request.session.clear()
            return None
        return {
            "email": email,
            "name": str(identity.get("name") or email),
            "subject": str(identity.get("subject") or ""),
            "auth_method": "microsoft",
        }

    def require_admin(self, request: Request) -> dict[str, str]:
        identity = self.current_admin(request)
        if identity is None:
            raise HTTPException(status_code=401, detail="Admin sign-in required.")
        return identity


ADMIN_AUTH = AdminAuth()
