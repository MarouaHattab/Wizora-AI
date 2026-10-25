from collections.abc import Mapping

from helper.asset_catalog import read_job
from helper.ingestion_errors import ProcessingError
from helper.job_results import cancel_job, complete_job, fail_job, start_job


def _setting(settings, key):
    return settings.get(key) if isinstance(settings, Mapping) else getattr(settings, key, None)


class WorkflowOperation:
    def __init__(
        self, workflow, state, settings, *, thread_id, request_id, refinement=False, action=None
    ):
        self.workflow, self.thread_id, self.request_id = workflow, thread_id, request_id
        self.settings = settings
        file_id = state.get("asset_id") if isinstance(state, Mapping) else None
        if not isinstance(file_id, str):
            raise ValueError("An operation requires its source file identifier")
        try:
            source = read_job(file_id)
        except ProcessingError:
            source = {}
        if workflow == "qa":
            job_type, prefix = "chatbot_response", "QUESTION_ANSWERER"
        else:
            accepted = (
                refinement
                and isinstance(action, str)
                and action.strip().casefold() in {"save", "accept"}
            )
            job_type = (
                "summary_acceptance"
                if accepted
                else "summarization_refinement"
                if refinement
                else "summarization"
            )
            prefix = "SUMMARIZER_REWRITER" if refinement and not accepted else "SUMMARIZER_WRITER"
        self.model_info = {
            "provider": _setting(settings, prefix + "_PROVIDER"),
            "model": _setting(settings, prefix + "_MODEL_ID"),
            "purpose": workflow,
        }
        config = {
            "action": "accept"
            if job_type == "summary_acceptance"
            else "refine"
            if refinement
            else "generate"
        }
        self.job_id = start_job(
            file_id,
            source.get("source_name", file_id),
            source.get("source_type", "unknown"),
            job_type,
            config,
            source_file_ids=source.get("source_file_ids") or [file_id],
            model_info=self.model_info,
        )
        from helper.provider_status import begin_provider_audit

        self.prefix = prefix
        self.audit_token = begin_provider_audit()

    def provider_details(self):
        from helper.provider_status import effective_model_info, provider_audit

        model = effective_model_info(self.prefix, self.settings)
        executions = provider_audit()
        model["executions"] = executions
        notices = list(
            dict.fromkeys(
                f"Configured fallback used: {entry['provider']} / {entry['model']}."
                for entry in executions
                if entry.get("fallback_used")
            )
        )
        return model, notices

    def finish(self, snapshot, *, status, error=None):
        from helper.provider_status import end_provider_audit

        model, warnings = self.provider_details()
        self.model_info.update(model)
        end_provider_audit(self.audit_token)
        values = getattr(snapshot, "values", None)
        values = values if isinstance(values, Mapping) else {}
        base = {"thread_id": self.thread_id, "run_id": self.request_id, "review_status": status}
        if self.workflow == "qa":
            from langchain_core.messages import AIMessage, HumanMessage

            from helper.workflow_history import canonical_messages

            messages = canonical_messages(values)
            answer = next(
                (item.content for item in reversed(messages) if isinstance(item, AIMessage)), ""
            )
            question = next(
                (item.content for item in reversed(messages) if isinstance(item, HumanMessage)), ""
            )
            results = {
                **base,
                "question": question,
                "answer": answer,
                "citations": values.get("answer_citations", []),
                "answer_status": values.get("answer_status"),
                "grounding_status": values.get("grounding_status"),
                "diagnostics": values.get("rag_diagnostics", {}),
            }
        else:
            results = {
                **base,
                "summary": values.get("summary", ""),
                "table_of_contents": values.get("Main_Points", ""),
                "citations": values.get("source_citations", []),
                "source_scope": values.get("source_scope", ""),
                "context_note": values.get("context_note", ""),
                "revision_notice": values.get("review_notice"),
            }
        if error is not None:
            from helper.sse import public_error

            return fail_job(
                self.job_id,
                public_error(error),
                results=results,
                model_info=self.model_info,
                warnings=warnings,
            )
        if status not in {"completed", "interrupted_for_feedback"}:
            return (
                cancel_job(self.job_id, model_info=self.model_info, warnings=warnings)
                if snapshot is None
                else fail_job(
                    self.job_id,
                    {
                        "category": "workflow",
                        "detail": "The operation did not produce a completed result.",
                    },
                    results=results,
                    model_info=self.model_info,
                    warnings=warnings,
                )
            )
        return complete_job(self.job_id, results, model_info=self.model_info, warnings=warnings)
