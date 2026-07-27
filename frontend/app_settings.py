"""Shared non-full-screen settings dialog for every AurumPilot page."""

from __future__ import annotations

import streamlit as st

from frontend.api_client import AurumPilotAPIError, api
from frontend.i18n import is_chinese, tr


def _credential_label(status: str) -> str:
    return tr({
        "CONFIGURED": "credential_configured",
        "NOT_CONFIGURED": "credential_not_configured",
        "DISABLED": "source_disabled",
    }.get(status, "credential_not_configured"))


def _source_copy(source_id: str, field: str, fallback: str) -> str:
    key = f"source_{source_id}_{field}"
    try:
        return tr(key)
    except KeyError:
        return fallback


def _apply_language(
    selector_key: str,
    chinese_label: str,
) -> None:
    """Update language in the widget callback and defer the app rerun."""
    selected_language = (
        "zh" if st.session_state[selector_key] == chinese_label else "en"
    )
    if selected_language != st.session_state.get("language", "zh"):
        st.session_state["language"] = selected_language
        st.session_state["settings_language_app_rerun_requested"] = True


def _render_credential_tools(
    *,
    configured_sources: list[dict],
    changed_source_count: int,
    editor_generation: int,
) -> None:
    if not configured_sources:
        return

    st.divider()
    with st.expander(tr("credential_tools"), expanded=False):
        st.caption(tr(
            "credential_tools_pending"
            if changed_source_count
            else "credential_tools_help"
        ))
        for source in configured_sources:
            source_id = source["source_id"]
            st.markdown(f"**{source['display_name']}**")
            if source_id == "databento":
                if st.button(
                    tr("verify_source"),
                    icon=":material/verified:",
                    disabled=changed_source_count > 0,
                    key=f"verify_source_databento_{editor_generation}",
                ):
                    try:
                        verification = api.post(
                            "settings/sources/databento/verify"
                        )
                        st.success(
                            f"{tr('source_verified')}: "
                            f"{verification['dataset']} · "
                            f"{verification['symbol']} · "
                            f"{verification['schema_name']}"
                        )
                    except AurumPilotAPIError:
                        st.error(
                            tr("source_verification_failed"),
                            icon=":material/error:",
                        )

                refresh_confirmed = st.checkbox(
                    tr("confirm_paid_refresh"),
                    key=(
                        "confirm_paid_databento_refresh_"
                        f"{editor_generation}"
                    ),
                )
                if st.button(
                    tr("refresh_market_data"),
                    icon=":material/sync:",
                    disabled=(
                        not refresh_confirmed
                        or changed_source_count > 0
                    ),
                    key=(
                        "refresh_databento_market_data_"
                        f"{editor_generation}"
                    ),
                ):
                    try:
                        api.post("market/refresh")
                        st.cache_data.clear()
                        st.toast(
                            tr("market_data_refreshed"),
                            icon=":material/check_circle:",
                        )
                        st.rerun()
                    except AurumPilotAPIError:
                        st.error(
                            tr("market_refresh_failed"),
                            icon=":material/error:",
                        )

            confirmed = st.checkbox(
                tr("confirm_key_delete"),
                key=(
                    f"confirm_delete_{source_id}_"
                    f"{editor_generation}"
                ),
            )
            if st.button(
                tr("delete_key"),
                icon=":material/delete:",
                disabled=(
                    not confirmed
                    or changed_source_count > 0
                ),
                key=f"delete_key_{source_id}_{editor_generation}",
            ):
                try:
                    api.delete(
                        f"settings/sources/{source_id}/credential"
                    )
                    st.session_state["source_editor_generation"] = (
                        editor_generation + 1
                    )
                    st.toast(
                        tr("key_deleted"),
                        icon=":material/check_circle:",
                    )
                    st.rerun()
                except AurumPilotAPIError:
                    st.error(
                        tr("operation_failed"),
                        icon=":material/error:",
                    )


def _render_sources_tab() -> None:
    try:
        payload = api.get("settings/sources")
    except AurumPilotAPIError:
        st.error(tr("connection_error"), icon=":material/cloud_off:")
        return

    if st.session_state.pop("source_settings_saved_notice", False):
        st.success(
            tr("source_settings_saved"),
            icon=":material/check_circle:",
        )

    st.warning(tr("plaintext_key_warning"), icon=":material/warning:")
    st.caption(f"{tr('secrets_file')}: `{payload['secrets_file_path']}`")
    st.caption(tr("source_settings_scope"))
    st.info(
        tr("macro_source_disable_semantics"),
        icon=":material/account_tree:",
    )

    databento_status = next(
        source["credential_status"]
        for source in payload["sources"]
        if source["source_id"] == "databento"
    )
    if not payload["market_data_ready"]:
        message_key = (
            "databento_refresh_required"
            if databento_status == "CONFIGURED"
            else "databento_setup_required"
        )
        st.info(tr(message_key), icon=":material/key:")

    editor_generation = int(
        st.session_state.get("source_editor_generation", 0)
    )
    source_updates: list[dict[str, object]] = []
    changed_source_count = 0

    for source in payload["sources"]:
        source_id = source["source_id"]
        with st.expander(
            source["display_name"],
            expanded=source_id == "databento",
        ):
            st.caption(_source_copy(
                source_id,
                "purpose",
                source["purpose"],
            ))
            st.caption(_source_copy(
                source_id,
                "cost",
                source["cost_notice"],
            ))
            st.badge(
                _credential_label(source["credential_status"]),
                color=(
                    "green"
                    if source["credential_status"] == "CONFIGURED"
                    else "orange"
                ),
            )
            enabled = st.toggle(
                tr("source_enabled"),
                value=source["enabled"],
                disabled=not source["can_disable"],
                key=f"source_enabled_{source_id}_{editor_generation}",
            )
            key_value = ""
            if source["credential_supported"]:
                key_value = st.text_input(
                    tr("api_key"),
                    type="password",
                    placeholder=(
                        tr("api_key_replace_hint")
                        if source["credential_status"] == "CONFIGURED"
                        else tr("api_key_input_hint")
                    ),
                    key=f"source_key_{source_id}_{editor_generation}",
                )

            normalized_key = key_value.strip()
            source_update: dict[str, object] = {
                "source_id": source_id,
                "enabled": enabled,
            }
            if normalized_key:
                source_update["api_key"] = normalized_key
            source_updates.append(source_update)
            if enabled != source["enabled"] or normalized_key:
                changed_source_count += 1

    if changed_source_count:
        st.info(
            tr("source_unsaved_changes").format(
                count=changed_source_count
            ),
            icon=":material/pending_actions:",
        )
    else:
        st.caption(tr("source_no_unsaved_changes"))

    if st.button(
        tr("save_all_sources"),
        icon=":material/save:",
        type="primary",
        disabled=changed_source_count == 0,
        width="stretch",
        key=f"save_all_sources_{editor_generation}",
    ):
        try:
            api.put(
                "settings/sources",
                {"sources": source_updates},
            )
            st.session_state["source_settings_saved_notice"] = True
            st.session_state["source_editor_generation"] = (
                editor_generation + 1
            )
            st.toast(
                tr("source_settings_saved"),
                icon=":material/check_circle:",
            )
            st.rerun()
        except AurumPilotAPIError:
            st.error(
                tr("source_settings_save_failed"),
                icon=":material/error:",
            )

    _render_credential_tools(
        configured_sources=[
            source
            for source in payload["sources"]
            if source["credential_supported"]
            and source["credential_status"] == "CONFIGURED"
        ],
        changed_source_count=changed_source_count,
        editor_generation=editor_generation,
    )


def _render_dialog() -> None:
    general_tab, sources_tab = st.tabs([
        tr("settings_general"),
        tr("settings_sources"),
    ])

    with general_tab:
        current_language = "zh" if is_chinese() else "en"
        labels = [tr("chinese"), tr("english")]
        selector_key = (
            f"settings_language_selector_{current_language}"
        )
        st.segmented_control(
            tr("language"),
            labels,
            default=(
                labels[0]
                if current_language == "zh"
                else labels[1]
            ),
            key=selector_key,
            width="stretch",
            on_change=_apply_language,
            args=(selector_key, labels[0]),
        )
        if st.session_state.pop(
            "settings_language_app_rerun_requested",
            False,
        ):
            st.rerun(scope="app")

    with sources_tab:
        _render_sources_tab()


@st.dialog("Settings", width="small")
def _settings_dialog_en() -> None:
    _render_dialog()


@st.dialog("设置", width="small")
def _settings_dialog_zh() -> None:
    _render_dialog()


def show_settings_dialog() -> None:
    """Open a localized modal; unsaved credential text stays in browser memory."""

    (_settings_dialog_zh if is_chinese() else _settings_dialog_en)()
