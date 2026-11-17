"""Route modul MDIK: dashboard, CRUD event, verifikasi, impor/ekspor, peta."""

from __future__ import annotations

import csv
import io
import math
import uuid
from datetime import datetime

from flask import (
    Blueprint, Response, flash, g, jsonify, redirect, render_template,
    request, session, url_for,
)

from .auth import ROLE_LABELS, authenticate, login_required, role_required
from .db import get_db, log_action, make_source_id, risk_from_height

bp = Blueprint("mdik", __name__, url_prefix="/mdik", template_folder="templates")

PAGE_SIZE = 25
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
RISK_ORDER = ["Awas", "Siaga", "Waspada", "Tidak diketahui"]
STATUS_LABELS = {
    "perlu_review": "Perlu review",
    "terverifikasi": "Terverifikasi",
    "ditolak": "Ditolak",
}


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def _num(value, cast=float, default=None):
    if value is None or str(value).strip() == "":
        return default
    try:
        number = cast(float(value))
        if isinstance(number, float) and math.isnan(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _pick(row: dict, *names):
    """Ambil nilai pertama yang ada dari beberapa alias kolom."""
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    return None


def _clean_text(value, max_len=300):
    if value is None:
        return None
    text = str(value).strip()
    return text[:max_len] if text else None


def _event_from_form(form, existing=None):
    """Validasi form -> dict siap simpan. Mengembalikan (data, errors)."""
    errors = []
    year = _num(form.get("year"), int)
    if year is None or not (-3000 <= year <= 2100):
        errors.append("Tahun wajib diisi dan berada di antara -3000 dan 2100.")
    country = _clean_text(form.get("country"), 120)
    if not country:
        errors.append("Negara/wilayah wajib diisi.")
    lat = _num(form.get("latitude"), float)
    lon = _num(form.get("longitude"), float)
    if lat is not None and not (-90 <= lat <= 90):
        errors.append("Latitude harus di antara -90 dan 90.")
    if lon is not None and not (-180 <= lon <= 180):
        errors.append("Longitude harus di antara -180 dan 180.")
    height = _num(form.get("max_height_m"), float)
    if height is not None and height < 0:
        errors.append("Tinggi air tidak boleh negatif.")
    for field, label in (("deaths", "Korban meninggal"), ("missing", "Hilang"),
                         ("injuries", "Luka")):
        value = _num(form.get(field), int)
        if value is not None and value < 0:
            errors.append(f"{label} tidak boleh negatif.")
    data = {
        "year": year,
        "month": _num(form.get("month"), int),
        "day": _num(form.get("day"), int),
        "country": country,
        "location": _clean_text(form.get("location")),
        "latitude": lat,
        "longitude": lon,
        "max_height_m": height,
        "eq_magnitude": _num(form.get("eq_magnitude"), float),
        "cause_code": _num(form.get("cause_code"), int),
        "validity": _num(form.get("validity"), int),
        "deaths": _num(form.get("deaths"), int),
        "missing": _num(form.get("missing"), int),
        "injuries": _num(form.get("injuries"), int),
        "damage_musd": _num(form.get("damage_musd"), float),
        "houses_destroyed": _num(form.get("houses_destroyed"), int),
        "houses_damaged": _num(form.get("houses_damaged"), int),
        "notes": _clean_text(form.get("notes"), 1000),
    }
    if existing:
        data["source_id"] = existing["source_id"]
    else:
        data["source_id"] = f"manual-{uuid.uuid4().hex[:12]}"
    data["risk_level"] = risk_from_height(height)
    return data, errors


def _query_events(filters, limit=None, offset=None):
    where, params = ["1=1"], []
    if filters.get("q"):
        where.append("(LOWER(country) LIKE ? OR LOWER(location) LIKE ?)")
        needle = f"%{filters['q'].lower()}%"
        params += [needle, needle]
    if filters.get("risk"):
        where.append("risk_level = ?")
        params.append(filters["risk"])
    if filters.get("status"):
        where.append("status = ?")
        params.append(filters["status"])
    if filters.get("decade"):
        where.append("year BETWEEN ? AND ?")
        params += [filters["decade"], filters["decade"] + 9]
    where_sql = " AND ".join(where)
    conn = get_db()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE " + where_sql, params
        ).fetchone()[0]
        sql = "SELECT * FROM events WHERE " + where_sql + " ORDER BY year DESC, id DESC"
        query_params = list(params)
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            query_params += [limit, offset or 0]
        rows = conn.execute(sql, query_params).fetchall()
    finally:
        conn.close()
    return rows, count


# --------------------------------------------------------------------------- #
# Dashboard & statistik
# --------------------------------------------------------------------------- #
def _dashboard_stats():
    conn = get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        by_status = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, COUNT(*) n FROM events GROUP BY status")}
        by_risk = {r["risk_level"]: r["n"] for r in conn.execute(
            "SELECT risk_level, COUNT(*) n FROM events GROUP BY risk_level")}
        countries = conn.execute(
            "SELECT COUNT(DISTINCT country) FROM events").fetchone()[0]
        with_height = conn.execute(
            "SELECT COUNT(*) FROM events WHERE max_height_m IS NOT NULL AND max_height_m > 0"
        ).fetchone()[0]
        with_coords = conn.execute(
            "SELECT COUNT(*) FROM events WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
        ).fetchone()[0]
        top_countries = conn.execute(
            "SELECT country, COUNT(*) n FROM events GROUP BY country "
            "ORDER BY n DESC LIMIT 8").fetchall()
        decades = conn.execute(
            "SELECT (year/10)*10 AS decade, COUNT(*) n FROM events "
            "WHERE year >= 1800 GROUP BY decade ORDER BY decade").fetchall()
        recent = conn.execute(
            "SELECT * FROM events ORDER BY year DESC, id DESC LIMIT 8").fetchall()
        imports = conn.execute(
            "SELECT * FROM imports ORDER BY id DESC LIMIT 5").fetchall()
        top_deaths = conn.execute(
            "SELECT * FROM events WHERE deaths IS NOT NULL "
            "ORDER BY deaths DESC LIMIT 5").fetchall()
    finally:
        conn.close()
    return {
        "total": total,
        "verified": by_status.get("terverifikasi", 0),
        "review": by_status.get("perlu_review", 0),
        "rejected": by_status.get("ditolak", 0),
        "countries": countries,
        "with_height": with_height,
        "with_coords": with_coords,
        "risk": {k: by_risk.get(k, 0) for k in RISK_ORDER},
        "top_countries": top_countries,
        "top_country_max": max([r["n"] for r in top_countries], default=1),
        "decades": decades,
        "decade_max": max([r["n"] for r in decades], default=1),
        "recent": recent,
        "imports": imports,
        "top_deaths": top_deaths,
    }


@bp.route("/")
def dashboard():
    stats = _dashboard_stats()
    risks = stats["risk"]
    total_height = sum(risks.values())
    risk_pct = {k: (v / total_height * 100 if total_height else 0) for k, v in risks.items()}
    return render_template(
        "mdik/dashboard.html", stats=stats, risk_pct=risk_pct,
        status_labels=STATUS_LABELS, active="dashboard",
    )


@bp.route("/api/stats")
def api_stats():
    stats = _dashboard_stats()
    for key in ("top_countries", "decades", "recent", "imports", "top_deaths"):
        stats[key] = [dict(row) for row in stats[key]]
    return jsonify(stats)


# --------------------------------------------------------------------------- #
# Daftar & detail event
# --------------------------------------------------------------------------- #
@bp.route("/events")
def events():
    page = max(_num(request.args.get("page"), int, 1), 1)
    filters = {
        "q": (request.args.get("q") or "").strip(),
        "risk": request.args.get("risk") or "",
        "status": request.args.get("status") or "",
        "decade": _num(request.args.get("decade"), int),
    }
    rows, total = _query_events(filters, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
    pages = max((total + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    return render_template(
        "mdik/events.html", events=rows, total=total, page=page, pages=pages,
        filters=filters, status_labels=STATUS_LABELS, risks=RISK_ORDER, active="events",
    )


@bp.route("/events/<int:event_id>")
def event_detail(event_id):
    conn = get_db()
    try:
        event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        audit = conn.execute(
            "SELECT * FROM audit_log WHERE entity = 'events' AND entity_id = ? "
            "ORDER BY id DESC LIMIT 10", (event_id,)).fetchall()
    finally:
        conn.close()
    if event is None:
        return render_template("mdik/404.html", active="events"), 404
    return render_template(
        "mdik/event_detail.html", event=event, audit=audit,
        status_labels=STATUS_LABELS, active="events",
    )


@bp.route("/events/new", methods=["GET", "POST"])
@login_required
@role_required("operator")
def event_new():
    if request.method == "POST":
        data, errors = _event_from_form(request.form)
        if errors:
            for error in errors:
                flash(error, "error")
            return render_template("mdik/event_form.html", event=request.form.to_dict(),
                                   mode="new", active="events"), 400
        conn = get_db()
        try:
            cur = conn.execute(
                """INSERT INTO events (source_id, year, month, day, country, location,
                   latitude, longitude, max_height_m, eq_magnitude, cause_code, validity,
                   deaths, missing, injuries, damage_musd, houses_destroyed, houses_damaged,
                   risk_level, status, notes)
                   VALUES (:source_id, :year, :month, :day, :country, :location,
                   :latitude, :longitude, :max_height_m, :eq_magnitude, :cause_code, :validity,
                   :deaths, :missing, :injuries, :damage_musd, :houses_destroyed,
                   :houses_damaged, :risk_level, 'perlu_review', :notes)""",
                data,
            )
            conn.commit()
            event_id = cur.lastrowid
        finally:
            conn.close()
        log_action(g.user["username"], "create", "events", event_id,
                   f"{data['country']} {data['year']}")
        flash("Event baru ditambahkan dan masuk antrean verifikasi.", "success")
        return redirect(url_for("mdik.event_detail", event_id=event_id))
    return render_template("mdik/event_form.html", event=None, mode="new", active="events")


@bp.route("/events/<int:event_id>/edit", methods=["GET", "POST"])
@login_required
@role_required("operator")
def event_edit(event_id):
    conn = get_db()
    try:
        event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    finally:
        conn.close()
    if event is None:
        return render_template("mdik/404.html", active="events"), 404
    if request.method == "POST":
        data, errors = _event_from_form(request.form, existing=event)
        if errors:
            for error in errors:
                flash(error, "error")
            return render_template("mdik/event_form.html", event=request.form.to_dict(),
                                   mode="edit", event_id=event_id, active="events"), 400
        conn = get_db()
        try:
            conn.execute(
                """UPDATE events SET year=:year, month=:month, day=:day, country=:country,
                   location=:location, latitude=:latitude, longitude=:longitude,
                   max_height_m=:max_height_m, eq_magnitude=:eq_magnitude,
                   cause_code=:cause_code, validity=:validity, deaths=:deaths,
                   missing=:missing, injuries=:injuries, damage_musd=:damage_musd,
                   houses_destroyed=:houses_destroyed, houses_damaged=:houses_damaged,
                   risk_level=:risk_level, notes=:notes, updated_at=datetime('now')
                   WHERE id=:id""",
                {**data, "id": event_id},
            )
            conn.commit()
        finally:
            conn.close()
        log_action(g.user["username"], "update", "events", event_id,
                   f"{data['country']} {data['year']}")
        flash("Perubahan disimpan.", "success")
        return redirect(url_for("mdik.event_detail", event_id=event_id))
    return render_template("mdik/event_form.html", event=dict(event), mode="edit",
                           event_id=event_id, active="events")


@bp.route("/events/<int:event_id>/delete", methods=["POST"])
@login_required
@role_required("operator")
def event_delete(event_id):
    conn = get_db()
    try:
        event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if event is None:
            return render_template("mdik/404.html", active="events"), 404
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        conn.commit()
    finally:
        conn.close()
    log_action(g.user["username"], "delete", "events", event_id,
               f"{event['country']} {event['year']}")
    flash("Event dihapus.", "success")
    return redirect(url_for("mdik.events"))


@bp.route("/events/<int:event_id>/verify", methods=["POST"])
@login_required
@role_required("analis")
def event_verify(event_id):
    action = request.form.get("action", "verify")
    status = "terverifikasi" if action == "verify" else "ditolak"
    conn = get_db()
    try:
        event = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if event is None:
            return render_template("mdik/404.html", active="events"), 404
        conn.execute(
            "UPDATE events SET status=?, verified_by=?, verified_at=datetime('now'), "
            "updated_at=datetime('now') WHERE id=?",
            (status, g.user["username"], event_id),
        )
        conn.commit()
    finally:
        conn.close()
    log_action(g.user["username"], action, "events", event_id,
               f"{event['country']} {event['year']} -> {status}")
    flash(f"Status event: {STATUS_LABELS[status]}.", "success")
    return redirect(url_for("mdik.event_detail", event_id=event_id))


# --------------------------------------------------------------------------- #
# Impor / ekspor
# --------------------------------------------------------------------------- #
@bp.route("/import", methods=["GET", "POST"])
@login_required
@role_required("operator")
def import_csv():
    if request.method == "POST":
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            flash("Pilih berkas CSV terlebih dahulu.", "error")
            return redirect(url_for("mdik.import_csv"))
        raw = upload.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            flash("Berkas melebihi batas 5 MB.", "error")
            return redirect(url_for("mdik.import_csv"))
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        if not reader.fieldnames:
            flash("Berkas tidak memiliki header CSV.", "error")
            return redirect(url_for("mdik.import_csv"))

        rows_total = rows_imported = rows_skipped = 0
        errors = []
        conn = get_db()
        try:
            for index, row in enumerate(reader, start=2):
                rows_total += 1
                year = _num(_pick(row, "Year", "year", "tahun"), int)
                if year is None or not (-3000 <= year <= 2100):
                    rows_skipped += 1
                    if len(errors) < 5:
                        errors.append(f"Baris {index}: tahun tidak valid.")
                    continue
                country = _clean_text(_pick(row, "Country", "country", "negara"), 120)
                if not country:
                    rows_skipped += 1
                    if len(errors) < 5:
                        errors.append(f"Baris {index}: negara kosong.")
                    continue
                lat = _num(_pick(row, "Latitude", "latitude", "lat"), float)
                lon = _num(_pick(row, "Longitude", "longitude", "lon", "lng"), float)
                height = _num(_pick(
                    row, "Maximum Water Height (m)", "max_height_m", "height_m",
                    "tinggi_air_m"), float)
                source_id = make_source_id("imp", repr(sorted(row.items())))
                payload = {
                    "source_id": source_id, "year": year,
                    "month": _num(_pick(row, "Mo", "month", "bulan"), int),
                    "day": _num(_pick(row, "Dy", "day", "tanggal"), int),
                    "country": country,
                    "location": _clean_text(_pick(row, "Location Name", "location", "lokasi")),
                    "latitude": lat, "longitude": lon, "max_height_m": height,
                    "eq_magnitude": _num(_pick(
                        row, "Earthquake Magnitude", "eq_magnitude", "magnitude"), float),
                    "cause_code": _num(_pick(row, "Tsunami Cause Code", "cause_code"), int),
                    "validity": _num(_pick(row, "Tsunami Event Validity", "validity"), int),
                    "deaths": _num(_pick(row, "Deaths", "deaths", "korban"), int),
                    "missing": _num(_pick(row, "Missing", "missing", "hilang"), int),
                    "injuries": _num(_pick(row, "Injuries", "injuries", "luka"), int),
                    "damage_musd": _num(_pick(row, "Damage ($Mil)", "damage_musd"), float),
                    "houses_destroyed": _num(_pick(
                        row, "Houses Destroyed", "houses_destroyed"), int),
                    "houses_damaged": _num(_pick(
                        row, "Houses Damaged", "houses_damaged"), int),
                    "risk_level": risk_from_height(height),
                }
                cur = conn.execute(
                    """INSERT OR IGNORE INTO events (
                       source_id, year, month, day, country, location, latitude, longitude,
                       max_height_m, eq_magnitude, cause_code, validity, deaths, missing,
                       injuries, damage_musd, houses_destroyed, houses_damaged, risk_level,
                       status) VALUES (
                       :source_id, :year, :month, :day, :country, :location, :latitude,
                       :longitude, :max_height_m, :eq_magnitude, :cause_code, :validity,
                       :deaths, :missing, :injuries, :damage_musd, :houses_destroyed,
                       :houses_damaged, :risk_level, 'perlu_review')""", payload)
                if cur.rowcount:
                    rows_imported += 1
                else:
                    rows_skipped += 1
            conn.execute(
                "INSERT INTO imports (username, filename, rows_total, rows_imported, "
                "rows_skipped) VALUES (?, ?, ?, ?, ?)",
                (g.user["username"], upload.filename, rows_total, rows_imported, rows_skipped),
            )
            conn.commit()
        finally:
            conn.close()
        log_action(g.user["username"], "import", "events", None,
                   f"{upload.filename}: {rows_imported}/{rows_total} baris")
        flash(f"Impor selesai: {rows_imported} baru, {rows_skipped} dilewati "
              f"dari {rows_total} baris.", "success" if rows_imported else "warning")
        for error in errors:
            flash(error, "warning")
        return redirect(url_for("mdik.events", status="perlu_review"))
    return render_template("mdik/import.html", active="import")


@bp.route("/export.csv")
def export_csv():
    filters = {
        "q": (request.args.get("q") or "").strip(),
        "risk": request.args.get("risk") or "",
        "status": request.args.get("status") or "",
        "decade": _num(request.args.get("decade"), int),
    }
    rows, _ = _query_events(filters)
    output = io.StringIO()
    fields = ["id", "source_id", "year", "month", "day", "country", "location",
              "latitude", "longitude", "max_height_m", "eq_magnitude", "risk_level",
              "status", "deaths", "missing", "injuries", "damage_musd", "notes"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    filename = f"mdik_events_{datetime.now():%Y%m%d_%H%M}.csv"
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


# --------------------------------------------------------------------------- #
# Peta, audit, autentikasi
# --------------------------------------------------------------------------- #
@bp.route("/map")
def map_view():
    return render_template("mdik/map.html", active="map")


@bp.route("/api/events.geojson")
def events_geojson():
    only_verified = not (g.get("user") and request.args.get("all") == "1")
    conn = get_db()
    try:
        sql = ("SELECT id, year, country, location, latitude, longitude, max_height_m, "
               "risk_level, status, deaths FROM events "
               "WHERE latitude IS NOT NULL AND longitude IS NOT NULL")
        if only_verified:
            sql += " AND status = 'terverifikasi'"
        sql += " ORDER BY year DESC LIMIT 3000"
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
        "properties": {k: r[k] for k in r.keys() if k not in ("latitude", "longitude")},
    } for r in rows]
    return jsonify({"type": "FeatureCollection", "features": features})


@bp.route("/audit")
@login_required
@role_required("analis")
def audit_log():
    page = max(_num(request.args.get("page"), int, 1), 1)
    conn = get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ? OFFSET ?",
            (100, (page - 1) * 100)).fetchall()
    finally:
        conn.close()
    return render_template("mdik/audit.html", rows=rows, total=total, page=page,
                           active="audit")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        user = authenticate(username, password)
        if user is None:
            flash("Nama pengguna atau kata sandi salah.", "error")
            return render_template("mdik/login.html", active="login"), 401
        session.clear()
        session["mdik_user_id"] = user["id"]
        session["mdik_role"] = user["role"]
        log_action(user["username"], "login", "users", user["id"])
        flash(f"Selamat datang, {user['full_name']}.", "success")
        next_url = request.args.get("next") or url_for("mdik.dashboard")
        if not next_url.startswith("/mdik"):
            next_url = url_for("mdik.dashboard")
        return redirect(next_url)
    return render_template("mdik/login.html", active="login")


@bp.route("/logout", methods=["POST"])
def logout():
    user = g.get("user")
    if user:
        log_action(user["username"], "logout", "users", user["id"])
    session.clear()
    flash("Anda telah keluar.", "success")
    return redirect(url_for("mdik.dashboard"))


@bp.app_context_processor
def inject_globals():
    return {"role_labels": ROLE_LABELS, "status_labels": STATUS_LABELS,
            "risk_order": RISK_ORDER}
