"""Describe observed module interfaces for smoke and probe feedback."""

from __future__ import annotations

import json

from execution.action_runtime.module_protocol import normalize_time_ranges


def _result(step: dict) -> dict:
    payload = step.get("module_result")
    if isinstance(payload, dict) and payload:
        return payload
    try:
        payload = json.loads(step.get("observation") or "{}")
    except (ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _successful(step: dict, payload: dict) -> bool:
    return (
        step.get("execution_status", "ok") == "ok"
        and payload.get("status") == "ok"
        and step.get("output_contract_valid") is not False
    )


def _consumption_matches(step: dict, event: dict) -> bool:
    if step.get("producer_module") == "video_structuring":
        source_ref = (step.get("tool_request") or {}).get("source_ref")
        return bool(source_ref and source_ref == event.get("source_ref"))
    event_ids = set(step.get("evidence_event_ids") or [])
    source_ids = set(event.get("source_event_ids") or [])
    if event_ids or source_ids:
        return bool(event_ids & source_ids)
    tool = step.get("action") or step.get("tool")
    return bool(tool and tool == event.get("tool_name"))


def _localized_input_received(result: dict, consumer_step: dict, fields: list[str]) -> bool:
    """Observe a successful perception request within the produced time ranges."""
    if not set(fields) <= {
        "time_ranges",
        "candidate_windows",
        "windows",
        "verified_windows",
        "ranges",
    }:
        return False
    if not _successful(consumer_step, _result(consumer_step)):
        return False
    requested = normalize_time_ranges((consumer_step.get("tool_request") or {}).get("params"))
    produced = normalize_time_ranges(result)
    return bool(requested and produced) and all(
        any(
            window["start_sec"] <= item["start_sec"] and item["end_sec"] <= window["end_sec"]
            for window in produced
        )
        for item in requested
    )


def evaluate_handoffs(trajectory: list, events: list, contracts: list) -> list[dict]:
    steps = [
        step
        for step in trajectory
        if isinstance(step, dict) and step.get("step_type") == "tool_execution"
    ]
    checks = []
    for contract in contracts:
        producer, consumer = contract["producer"], contract["consumer"]
        fields = list(contract.get("required_fields") or [])
        consumption = [
            event
            for event in events
            if isinstance(event, dict)
            and event.get("event") == "consumer_consumed"
            and event.get("status") == "ok"
            and event.get("contract_valid") is True
            and event.get("producer_module") == producer
            and event.get("consumer") == consumer
        ]
        witnesses = []
        for index, step in enumerate(steps):
            if step.get("producer_module") != producer:
                continue
            payload = _result(step)
            evidence = payload.get("evidence")
            available = set(payload) | (set(evidence) if isinstance(evidence, dict) else set())
            linked = [event for event in consumption if _consumption_matches(step, event)]
            localized_input = (
                producer == "localization"
                and consumer == "perception"
                and any(
                    item.get("producer_module") == consumer
                    and _localized_input_received(payload, item, fields)
                    for item in steps[index + 1 :]
                )
            )
            witnesses.append(
                {
                    "missing_fields": [field for field in fields if field not in available],
                    "producer_success": _successful(step, payload),
                    "consumer_observed": bool(linked) or localized_input,
                    "consumption_event": bool(linked),
                    "delivery_evidence": "runtime_event"
                    if linked
                    else "localized_tool_input"
                    if localized_input
                    else "",
                    "payload_parseable": bool(payload),
                }
            )
        if producer == "memory":
            witnesses.extend(
                {
                    "missing_fields": [
                        field for field in fields if field not in (event.get("output_fields") or [])
                    ],
                    "consumer_observed": True,
                    "producer_success": True,
                    "consumption_event": True,
                    "delivery_evidence": "runtime_event",
                    "payload_parseable": True,
                }
                for event in consumption
            )
        valid = [
            item
            for item in witnesses
            if item["producer_success"] and not item["missing_fields"] and item["consumer_observed"]
        ]
        witness = next(iter(valid or witnesses), {})
        status = "observed" if valid else "incomplete" if witnesses else "not_exercised"
        checks.append(
            {
                "producer": producer,
                "consumer": consumer,
                "required_fields": fields,
                "producer_observed": bool(witnesses),
                "producer_success": witness.get("producer_success", False),
                "status": status,
                "missing_fields": witness.get("missing_fields", []),
                "consumer_observed": witness.get("consumer_observed", False),
                "consumption_event": witness.get("consumption_event", False),
                "delivery_evidence": witness.get("delivery_evidence", ""),
                "payload_parseable": witness.get("payload_parseable", False),
                "passed": status != "incomplete",
            }
        )
    return checks
