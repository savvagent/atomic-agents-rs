//! Static validation of a [`Pipeline`], run before anything executes.

use std::collections::BTreeSet;

use crate::condition::{check_bounds, parse};
use crate::spec::{
    Condition, MAX_FOREACH_ITEMS, MAX_MAX_RETRIES, MAX_MAX_TURNS, Pipeline, Step, StepKind,
};

/// Checks every rule serde cannot; returns all violations, each prefixed with
/// a JSON pointer such as `/steps/1/max_items`.
///
/// ```
/// use atomic_agent_runner::{spec::Pipeline, validate::validate};
/// let p: Pipeline = serde_json::from_value(serde_json::json!(
///     {"spec_version": 2, "name": "n", "steps": []})).unwrap();
/// assert_eq!(validate(&p).unwrap_err()[0], "/spec_version: only version 1 is supported");
/// ```
pub fn validate(p: &Pipeline) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();
    if p.spec_version != 1 {
        errs.push("/spec_version: only version 1 is supported".to_string());
    }
    if let Some(r) = p.max_retries
        && r > MAX_MAX_RETRIES
    {
        errs.push(format!("/max_retries: must be in 0..={MAX_MAX_RETRIES}"));
    }
    let mut reserved: BTreeSet<String> = ["event", "state"].map(String::from).into();
    collect_as_names(&p.steps, &mut reserved);
    let mut ctx = Ctx {
        errs,
        reserved,
        seen: BTreeSet::new(),
        pipeline_model: p.model.is_some(),
    };
    ctx.steps(&p.steps, "/steps", false);
    if ctx.errs.is_empty() {
        Ok(())
    } else {
        Err(ctx.errs)
    }
}

fn collect_as_names(steps: &[Step], out: &mut BTreeSet<String>) {
    for s in steps {
        if let StepKind::Foreach(f) = &s.kind {
            out.insert(f.item_name().to_string());
            collect_as_names(&f.steps, out);
        }
    }
}

struct Ctx {
    errs: Vec<String>,
    /// `event`, `state` and every foreach item name.
    reserved: BTreeSet<String>,
    seen: BTreeSet<String>,
    pipeline_model: bool,
}

fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| matches!(c, 'a'..='z' | '0'..='9' | '_' | '-'))
}

impl Ctx {
    fn err(&mut self, path: &str, msg: impl std::fmt::Display) {
        self.errs.push(format!("{path}: {msg}"));
    }

    fn name(&mut self, path: &str, what: &str, name: &str) {
        if !valid_name(name) {
            self.err(
                path,
                format!("{what} must be non-empty and use only [a-z0-9_-]"),
            );
        }
    }

    fn steps(&mut self, steps: &[Step], base: &str, nested: bool) {
        for (i, step) in steps.iter().enumerate() {
            self.step(step, &format!("{base}/{i}"), nested);
        }
    }

    fn step(&mut self, step: &Step, at: &str, nested: bool) {
        let id_path = format!("{at}/id");
        self.name(&id_path, "id", &step.id);
        if self.reserved.contains(&step.id) {
            self.err(
                &id_path,
                format!(
                    "'{}' is reserved (event, state or a foreach item name)",
                    step.id
                ),
            );
        }
        if !self.seen.insert(step.id.clone()) {
            self.err(&id_path, format!("duplicate id '{}'", step.id));
        }
        let when_err = match &step.when {
            Some(Condition::Expr(src)) => parse(src).err(),
            Some(Condition::Structured(expr)) => check_bounds(expr).err(),
            None => None,
        };
        if let Some(e) = when_err {
            self.err(&format!("{at}/when"), e);
        }
        if step.extras.contains_key("constraints") && !matches!(step.kind, StepKind::Act(_)) {
            self.not_act(at);
        }
        match &step.kind {
            StepKind::Agent(a) => {
                if a.model.is_none() && !self.pipeline_model {
                    self.err(
                        &format!("{at}/model"),
                        "agent step needs a model (set it on the step or the pipeline)",
                    );
                }
                if let Some(t) = a.max_turns
                    && !(1..=MAX_MAX_TURNS).contains(&t)
                {
                    self.err(
                        &format!("{at}/max_turns"),
                        format!("must be in 1..={MAX_MAX_TURNS}"),
                    );
                }
                if let Some(r) = a.max_retries
                    && r > MAX_MAX_RETRIES
                {
                    self.err(
                        &format!("{at}/max_retries"),
                        format!("must be in 0..={MAX_MAX_RETRIES}"),
                    );
                }
                if let Err(e) = jsonschema::validator_for(&a.output_schema) {
                    self.err(
                        &format!("{at}/output_schema"),
                        format!("invalid JSON Schema: {e}"),
                    );
                }
            }
            StepKind::Tool(_) => {}
            StepKind::Act(a) => {
                // The spec fixes an act's argument keys; only values may be
                // templates, so event or model data cannot add arguments.
                if !a.args.is_object() {
                    self.err(
                        &format!("{at}/args"),
                        "act arguments must be a JSON object (values may be templates)",
                    );
                }
            }
            StepKind::Foreach(f) => {
                if nested {
                    self.err(at, "foreach cannot be nested inside another foreach");
                }
                if !(1..=MAX_FOREACH_ITEMS).contains(&f.max_items) {
                    self.err(
                        &format!("{at}/max_items"),
                        format!("must be in 1..={MAX_FOREACH_ITEMS}"),
                    );
                }
                if let Some(name) = &f.as_ {
                    self.name(&format!("{at}/as"), "as", name);
                    if name == "event" || name == "state" {
                        self.err(&format!("{at}/as"), format!("'{name}' is reserved"));
                    }
                }
                self.steps(&f.steps, &format!("{at}/steps"), true);
            }
        }
    }

    fn not_act(&mut self, at: &str) {
        self.err(
            &format!("{at}/constraints"),
            "constraints are only allowed on act steps",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn check(v: Value) -> Result<(), Vec<String>> {
        validate(&serde_json::from_value::<Pipeline>(v).unwrap())
    }

    fn errs(v: Value) -> Vec<String> {
        check(v).unwrap_err()
    }

    fn has(errs: &[String], needle: &str) -> bool {
        errs.iter().any(|e| e.contains(needle))
    }

    fn tool(id: &str) -> Value {
        json!({"id": id, "kind": "tool", "tool": "t", "args": {}})
    }

    fn agent(id: &str) -> Value {
        json!({"id": id, "kind": "agent", "system_prompt": "p", "input": "i",
               "output_schema": {"type": "object"}})
    }

    fn pipe(steps: Vec<Value>) -> Value {
        json!({"spec_version": 1, "name": "n", "model": "m", "steps": steps})
    }

    #[test]
    fn accepts_a_good_pipeline() {
        let f = json!({"id": "f", "kind": "foreach", "items": "a.xs", "max_items": 5,
                       "steps": [tool("g")]});
        let mut act = tool("c");
        act["kind"] = json!("act");
        act["constraints"] = json!({"n": {"lte": 3.0}});
        act["when"] = json!("a.ok && !a.bad");
        assert_eq!(check(pipe(vec![agent("a"), f, act])), Ok(()));
    }

    #[test]
    fn spec_version() {
        let mut p = pipe(vec![]);
        p["spec_version"] = json!(2);
        assert!(has(&errs(p), "/spec_version"));
    }

    #[test]
    fn ids() {
        let e = errs(pipe(vec![
            tool(""),
            tool("Bad Id"),
            tool("dup"),
            tool("dup"),
        ]));
        assert!(has(&e, "/steps/0/id"));
        assert!(has(&e, "/steps/1/id"));
        assert!(has(&e, "/steps/3/id: duplicate"));
        let e = errs(pipe(vec![tool("event"), tool("state"), tool("item")]));
        assert_eq!(e.len(), 2, "{e:?}"); // `item` is reserved only when a foreach uses it
    }

    #[test]
    fn ids_unique_across_levels_and_not_as_name() {
        let f = json!({"id": "f", "kind": "foreach", "items": "x", "as": "row", "max_items": 1,
                       "steps": [tool("a"), tool("row")]});
        let e = errs(pipe(vec![tool("a"), f]));
        assert!(has(&e, "/steps/1/steps/0/id: duplicate"), "{e:?}");
        assert!(
            has(&e, "/steps/1/steps/1/id") && has(&e, "reserved"),
            "{e:?}"
        );
    }

    #[test]
    fn foreach_rules() {
        let mk = |max: u64, steps: Vec<Value>| json!({"id": "f", "kind": "foreach", "items": "x", "max_items": max, "steps": steps});
        assert!(has(&errs(pipe(vec![mk(0, vec![])])), "/steps/0/max_items"));
        assert!(has(
            &errs(pipe(vec![mk(101, vec![])])),
            "/steps/0/max_items"
        ));
        assert_eq!(check(pipe(vec![mk(100, vec![])])), Ok(()));
        let inner =
            json!({"id": "in", "kind": "foreach", "items": "y", "max_items": 1, "steps": []});
        let e = errs(pipe(vec![mk(1, vec![inner])]));
        assert!(
            has(&e, "/steps/0/steps/0: foreach cannot be nested"),
            "{e:?}"
        );
        let mut bad_as = mk(1, vec![]);
        bad_as["as"] = json!("state");
        assert!(has(&errs(pipe(vec![bad_as])), "/steps/0/as"));
    }

    #[test]
    fn agent_limits_schema_and_model() {
        let mut a = agent("a");
        a["max_turns"] = json!(0);
        a["max_retries"] = json!(6);
        a["output_schema"] = json!({"type": "nonsense"});
        let e = errs(pipe(vec![a]));
        assert!(has(&e, "/steps/0/max_turns"));
        assert!(has(&e, "/steps/0/max_retries"));
        assert!(has(&e, "/steps/0/output_schema"));
        let mut a = agent("a");
        a["max_turns"] = json!(33);
        assert!(has(&errs(pipe(vec![a])), "/steps/0/max_turns"));
        let mut ok = agent("a");
        ok["max_turns"] = json!(32);
        ok["max_retries"] = json!(0);
        assert_eq!(check(pipe(vec![ok])), Ok(()));

        let mut no_model = pipe(vec![agent("a")]);
        no_model.as_object_mut().unwrap().remove("model");
        assert!(has(&errs(no_model.clone()), "/steps/0/model"));
        no_model["steps"][0]["model"] = json!("own");
        assert_eq!(check(no_model), Ok(()));
    }

    #[test]
    fn pipeline_max_retries_bound() {
        let mut p = pipe(vec![]);
        p["max_retries"] = json!(9);
        assert!(has(&errs(p), "/max_retries"));
    }

    #[test]
    fn when_strings_must_parse() {
        let mut t = tool("a");
        t["when"] = json!("a ==");
        let e = errs(pipe(vec![t]));
        assert!(has(&e, "/steps/0/when"), "{e:?}");
        let mut t = tool("a");
        t["when"] = json!({"op": "exists", "field": "x"});
        assert_eq!(check(pipe(vec![t])), Ok(()));
    }

    #[test]
    fn act_constraints_accepted() {
        let mut act = tool("c");
        act["kind"] = json!("act");
        act["constraints"] = json!({"n": {"lt": 1.0}});
        assert_eq!(check(pipe(vec![act])), Ok(()));
    }

    #[test]
    fn structured_conditions_are_bounded() {
        let mut deep = json!({"op": "exists", "field": "a"});
        for _ in 0..40 {
            deep = json!({"op": "not", "condition": deep});
        }
        let mut t = tool("a");
        t["when"] = deep;
        assert!(has(&errs(pipe(vec![t])), "/steps/0/when"));
        let mut t = tool("a");
        t["when"] = json!({"op": "matches", "field": "x", "pattern": "("});
        assert!(has(&errs(pipe(vec![t])), "/steps/0/when"));
        let mut t = tool("a");
        t["when"] = json!({"op": "matches", "field": "x", "pattern": "a".repeat(300)});
        assert!(has(&errs(pipe(vec![t])), "/steps/0/when"));
        let mut t = tool("a");
        t["when"] = json!({"op": "in", "field": "x", "values": vec![1; 101]});
        assert!(has(&errs(pipe(vec![t])), "/steps/0/when"));
    }

    #[test]
    fn remote_ref_in_output_schema_is_never_fetched() {
        // With the resolver features off, validation must fail fast (no network).
        let mut a = agent("a");
        a["output_schema"] = json!({"$ref": "https://example.invalid/schema.json"});
        assert!(has(&errs(pipe(vec![a])), "/steps/0/output_schema"));
    }

    #[test]
    fn max_retries_messages_are_consistent() {
        let mut p = pipe(vec![]);
        p["max_retries"] = json!(9);
        let mut a = agent("a");
        a["max_retries"] = json!(9);
        let pe = errs(p).remove(0);
        let ae = errs(pipe(vec![a])).remove(0);
        assert!(pe.ends_with("must be in 0..=5"), "{pe}");
        assert!(ae.ends_with("must be in 0..=5"), "{ae}");
    }

    #[test]
    fn act_args_must_be_a_literal_object() {
        let act = |args: Value| json!({"id": "c", "kind": "act", "tool": "w", "args": args});
        assert_eq!(
            check(pipe(vec![act(json!({"label": "{{ a.label }}", "n": 1}))])),
            Ok(())
        );
        for bad in [json!("{{triage}}"), json!(["x"]), json!(3)] {
            let e = errs(pipe(vec![act(bad.clone())]));
            assert!(
                has(&e, "/steps/0/args: act arguments must be a JSON object"),
                "{bad}: {e:?}"
            );
        }
        let f = json!({"id": "f", "kind": "foreach", "items": "x", "max_items": 1,
                       "steps": [act(json!("{{ item }}"))]});
        assert!(has(&errs(pipe(vec![f])), "/steps/0/steps/0/args"));
        // A tool step's arguments may still be one template.
        let mut t = tool("t");
        t["args"] = json!("{{ event }}");
        assert_eq!(check(pipe(vec![t])), Ok(()));
    }

    #[test]
    fn constraints_only_on_act() {
        let c = json!({"n": {"lt": 1.0}});
        let mut t = tool("t");
        t["constraints"] = c.clone();
        let mut a = agent("a");
        a["constraints"] = c.clone();
        let mut f =
            json!({"id": "f", "kind": "foreach", "items": "x", "max_items": 1, "steps": []});
        f["constraints"] = c;
        let e = errs(pipe(vec![t, a, f]));
        for i in 0..3 {
            assert!(has(&e, &format!("/steps/{i}/constraints")), "{e:?}");
        }
    }
}
