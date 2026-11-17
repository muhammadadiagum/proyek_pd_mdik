"""Modul MDIK — manajemen data & informasi kebencanaan."""

from flask import current_app, g

from . import auth, db, routes


def init_app(app):
    """Daftarkan blueprint MDIK, init DB, hook auth."""
    app.register_blueprint(routes.bp)
    app.before_request(auth.load_logged_in_user)
    db.init_db()
    app.logger.info("MDIK siap: %s", db.DB_PATH)


__all__ = ["init_app"]
