"""Autentikasi + otorisasi berbasis peran untuk modul MDIK."""

from functools import wraps

from flask import flash, g, redirect, request, session, url_for
from werkzeug.security import check_password_hash

from .db import get_db

ROLE_LABELS = {
    "admin": "Admin BPBD",
    "operator": "Operator Data",
    "analis": "Analis Kebencanaan",
    "publik": "Publik",
}


def load_logged_in_user():
    """Hook before_request: isi g.user dari session."""
    user_id = session.get("mdik_user_id")
    if user_id is None:
        g.user = None
        return
    conn = get_db()
    try:
        g.user = conn.execute(
            "SELECT id, username, full_name, role FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    finally:
        conn.close()


def authenticate(username: str, password: str):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    finally:
        conn.close()
    if row and check_password_hash(row["password_hash"], password):
        return row
    return None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.get("user") is None:
            flash("Silakan masuk untuk mengelola data.", "warning")
            return redirect(url_for("mdik.login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def role_required(*roles):
    """Batasi route ke peran tertentu; admin selalu boleh."""
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = g.get("user")
            if user is None:
                flash("Silakan masuk terlebih dahulu.", "warning")
                return redirect(url_for("mdik.login", next=request.path))
            if user["role"] != "admin" and user["role"] not in roles:
                flash("Peran Anda tidak berwenang untuk aksi ini.", "error")
                return redirect(url_for("mdik.dashboard"))
            return view(*args, **kwargs)

        return wrapped

    return decorator
