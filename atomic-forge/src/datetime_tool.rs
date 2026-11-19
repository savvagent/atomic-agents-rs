use atomic_agents::prelude::*;
use chrono::{
    DateTime, Datelike, Duration, FixedOffset, LocalResult, NaiveDate, NaiveDateTime, Offset,
    TimeZone, Timelike, Utc,
};
use chrono_tz::Tz;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The date/time operation to perform.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DateTimeOperation {
    /// Return the current time.
    Now,
    /// Parse an input datetime string.
    Parse,
    /// Convert a datetime from one timezone to another.
    Convert,
    /// Add a duration to a datetime.
    Shift,
    /// Compute the time difference between two datetimes.
    Diff,
}

impl DateTimeOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Now => "now",
            Self::Parse => "parse",
            Self::Convert => "convert",
            Self::Shift => "shift",
            Self::Diff => "diff",
        }
    }
}

fn default_timezone() -> String {
    "UTC".to_string()
}

/// Tool for date and time operations: getting the current time in any timezone,
/// parsing datetime strings, converting between timezones, shifting a datetime by a
/// duration, and computing the difference between two datetimes. Use this whenever
/// you need to know "what is today's date", convert a meeting time across regions,
/// or do any temporal arithmetic.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct DateTimeToolInput {
    /// The operation to perform. 'now' returns the current time. 'parse' parses
    /// an input datetime string. 'convert' converts a datetime from one timezone
    /// to another. 'shift' adds a duration to a datetime. 'diff' computes the
    /// time difference between two datetimes.
    pub operation: DateTimeOperation,
    /// IANA timezone name to interpret the input/result in (e.g. 'UTC',
    /// 'America/New_York', 'Europe/Brussels', 'Asia/Tokyo'). For 'convert' this
    /// is the source timezone of `input`.
    #[serde(default = "default_timezone")]
    pub timezone: String,
    /// Target IANA timezone for the 'convert' operation.
    #[serde(default)]
    pub target_timezone: Option<String>,
    /// Datetime string for 'parse', 'convert', and 'shift' operations. ISO 8601
    /// is preferred (e.g. '2026-04-29T15:30:00'). If the string has no timezone
    /// info, `timezone` is used.
    #[serde(default)]
    pub input: Option<String>,
    /// Second datetime string for the 'diff' operation (computes input_2 - input).
    #[serde(default)]
    pub input_2: Option<String>,
    /// Calendar days to add for 'shift' (negative to subtract), before elapsed
    /// hours/minutes/seconds.
    #[serde(default)]
    pub days: i64,
    /// Elapsed hours to add for 'shift' (negative to subtract).
    #[serde(default)]
    pub hours: i64,
    /// Elapsed minutes to add for 'shift' (negative to subtract).
    #[serde(default)]
    pub minutes: i64,
    /// Elapsed seconds to add for 'shift' (negative to subtract).
    #[serde(default)]
    pub seconds: i64,
}

impl DateTimeToolInput {
    /// Create an input for the given operation with all other fields defaulted.
    pub fn new(operation: DateTimeOperation) -> Self {
        Self {
            operation,
            timezone: default_timezone(),
            target_timezone: None,
            input: None,
            input_2: None,
            days: 0,
            hours: 0,
            minutes: 0,
            seconds: 0,
        }
    }
}

/// Output of the DateTimeTool. `error` is populated only when the operation fails.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, Default, PartialEq)]
pub struct DateTimeToolOutput {
    /// The operation that was performed.
    pub operation: String,
    /// ISO 8601 representation of the resulting datetime.
    pub iso: Option<String>,
    /// IANA timezone of the resulting datetime.
    pub timezone: Option<String>,
    /// POSIX/Unix timestamp (seconds since epoch).
    pub unix_timestamp: Option<f64>,
    /// Year component of the resulting datetime.
    pub year: Option<i32>,
    /// Month component (1-12).
    pub month: Option<u32>,
    /// Day-of-month component (1-31).
    pub day: Option<u32>,
    /// Hour component (0-23).
    pub hour: Option<u32>,
    /// Minute component (0-59).
    pub minute: Option<u32>,
    /// Second component (0-59).
    pub second: Option<u32>,
    /// Day name, e.g. 'Monday'.
    pub weekday: Option<String>,
    /// Human-readable formatted datetime, e.g. 'Wed, 29 Apr 2026 15:30 UTC'.
    pub human: Option<String>,
    /// Total seconds between input_2 and input (for 'diff').
    pub diff_seconds: Option<f64>,
    /// Human-readable difference, e.g. '2 days, 3 hours, 15 minutes'.
    pub diff_human: Option<String>,
    /// Error message if the operation failed.
    pub error: Option<String>,
}

/// Configuration for the DateTimeTool. Currently has no tunable settings beyond
/// title/description overrides.
pub struct DateTimeToolConfig {
    pub component: ComponentConfig,
}

impl Default for DateTimeToolConfig {
    fn default() -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("DateTime")
                .with_description("Date and time operations: now, parse, convert, shift, diff"),
        }
    }
}

/// A timezone: either a named IANA zone or a fixed UTC offset parsed from input.
#[derive(Clone, Copy, Debug)]
enum Zone {
    Named(Tz),
    Fixed(FixedOffset),
}

impl Zone {
    fn named(name: &str) -> std::result::Result<Self, String> {
        name.parse::<Tz>().map(Zone::Named).map_err(|_| {
            format!("Unknown timezone '{name}'. Use an IANA name like 'UTC' or 'America/New_York'.")
        })
    }

    fn offset_at(&self, utc: &DateTime<Utc>) -> FixedOffset {
        match self {
            Zone::Named(tz) => tz.offset_from_utc_datetime(&utc.naive_utc()).fix(),
            Zone::Fixed(fo) => *fo,
        }
    }

    /// Short zone name as shown in `human` (e.g. "CEST", "UTC", "UTC+02:00").
    fn abbreviation(&self, utc: &DateTime<Utc>) -> String {
        match self {
            Zone::Named(tz) => utc.with_timezone(tz).format("%Z").to_string(),
            Zone::Fixed(fo) => fixed_name(*fo),
        }
    }

    fn name(&self, utc: &DateTime<Utc>) -> String {
        match self {
            Zone::Named(tz) => tz.name().to_string(),
            Zone::Fixed(_) => self.abbreviation(utc),
        }
    }

    /// Resolve a local wall-clock time to an instant. An overlap resolves to the
    /// earlier instant. `None` means the local time falls in a DST gap.
    fn resolve_local(&self, naive: NaiveDateTime) -> Option<DateTime<Utc>> {
        match self {
            Zone::Named(tz) => match tz.from_local_datetime(&naive) {
                LocalResult::Single(dt) => Some(dt.with_timezone(&Utc)),
                LocalResult::Ambiguous(first, _) => Some(first.with_timezone(&Utc)),
                LocalResult::None => None,
            },
            Zone::Fixed(fo) => fo
                .from_local_datetime(&naive)
                .single()
                .map(|d| d.with_timezone(&Utc)),
        }
    }

    /// Like [`Zone::resolve_local`], but a time inside a DST gap is read with the
    /// offset in force before the gap, then normalised to its real instant.
    fn resolve_local_lenient(&self, naive: NaiveDateTime) -> Option<DateTime<Utc>> {
        if let Some(utc) = self.resolve_local(naive) {
            return Some(utc);
        }
        let Zone::Named(tz) = self else { return None };
        let probe = naive.checked_sub_signed(Duration::hours(24))?;
        let before = tz.from_local_datetime(&probe).earliest()?.offset().fix();
        let utc = naive.checked_sub_signed(Duration::seconds(i64::from(before.local_minus_utc())))?;
        Some(Utc.from_utc_datetime(&utc))
    }
}

fn fixed_name(fo: FixedOffset) -> String {
    let secs = fo.local_minus_utc();
    if secs == 0 {
        return "UTC".to_string();
    }
    let sign = if secs < 0 { '-' } else { '+' };
    let abs = secs.abs();
    let (h, m, s) = (abs / 3600, (abs % 3600) / 60, abs % 60);
    if s == 0 {
        format!("UTC{sign}{h:02}:{m:02}")
    } else {
        format!("UTC{sign}{h:02}:{m:02}:{s:02}")
    }
}

/// An instant together with the zone it is displayed in.
#[derive(Clone, Copy, Debug)]
struct Moment {
    utc: DateTime<Utc>,
    zone: Zone,
}

impl Moment {
    fn local(&self) -> NaiveDateTime {
        let off = i64::from(self.zone.offset_at(&self.utc).local_minus_utc());
        self.utc.naive_utc() + Duration::seconds(off)
    }

    fn build(&self, operation: &str) -> DateTimeToolOutput {
        let local = self.local();
        let offset = self.zone.offset_at(&self.utc);
        let micros = local.nanosecond() / 1_000;
        let frac = if micros == 0 {
            String::new()
        } else {
            format!(".{micros:06}")
        };
        let iso = format!(
            "{}{frac}{}",
            local.format("%Y-%m-%dT%H:%M:%S"),
            offset_iso(offset)
        );
        let ts = self.utc.timestamp() as f64 + f64::from(self.utc.timestamp_subsec_micros()) / 1e6;
        DateTimeToolOutput {
            operation: operation.to_string(),
            iso: Some(iso),
            timezone: Some(self.zone.name(&self.utc)),
            unix_timestamp: Some(ts),
            year: Some(local.year()),
            month: Some(local.month()),
            day: Some(local.day()),
            hour: Some(local.hour()),
            minute: Some(local.minute()),
            second: Some(local.second()),
            weekday: Some(local.format("%A").to_string()),
            human: Some(format!(
                "{} {}",
                local.format("%a, %d %b %Y %H:%M"),
                self.zone.abbreviation(&self.utc)
            )),
            ..Default::default()
        }
    }
}

fn offset_iso(offset: FixedOffset) -> String {
    let secs = offset.local_minus_utc();
    let sign = if secs < 0 { '-' } else { '+' };
    let abs = secs.abs();
    let (h, m, s) = (abs / 3600, (abs % 3600) / 60, abs % 60);
    if s == 0 {
        format!("{sign}{h:02}:{m:02}")
    } else {
        format!("{sign}{h:02}:{m:02}:{s:02}")
    }
}

/// Date and time operations: now, parse, convert, shift, diff.
///
/// # Example
///
/// ```
/// use atomic_agents::prelude::*;
/// use atomic_forge::datetime_tool::*;
///
/// let tool = DateTimeTool::new(DateTimeToolConfig::default());
/// let mut input = DateTimeToolInput::new(DateTimeOperation::Convert);
/// input.input = Some("2026-04-29T15:30:00".into());
/// input.timezone = "Europe/Brussels".into();
/// input.target_timezone = Some("America/New_York".into());
/// let out = tool.run(input).unwrap();
/// assert_eq!(out.iso.as_deref(), Some("2026-04-29T09:30:00-04:00"));
/// ```
pub struct DateTimeTool {
    config: ComponentConfig,
}

impl DateTimeTool {
    pub fn new(config: DateTimeToolConfig) -> Self {
        Self {
            config: config.component,
        }
    }

    /// Parse an ISO 8601 datetime string. A trailing 'Z' is UTC. If the value has
    /// no offset, `default_tz` is attached. A date-only value means midnight.
    fn parse(value: &str, default_tz: &str) -> std::result::Result<Moment, String> {
        let text = value.trim();
        let text = match text.strip_suffix('Z') {
            Some(rest) => format!("{rest}+00:00"),
            None => text.to_string(),
        };

        if let Ok(dt) = DateTime::parse_from_rfc3339(&text) {
            return Ok(Moment {
                utc: dt.with_timezone(&Utc),
                zone: Zone::Fixed(*dt.offset()),
            });
        }
        const OFFSET_FORMATS: [&str; 8] = [
            "%Y-%m-%dT%H:%M:%S%.f%:z",
            "%Y-%m-%d %H:%M:%S%.f%:z",
            "%Y-%m-%dT%H:%M%:z",
            "%Y-%m-%d %H:%M%:z",
            "%Y-%m-%dT%H:%M:%S%.f%z",
            "%Y-%m-%d %H:%M:%S%.f%z",
            "%Y-%m-%dT%H:%M%z",
            "%Y-%m-%d %H:%M%z",
        ];
        for fmt in OFFSET_FORMATS {
            if let Ok(dt) = DateTime::parse_from_str(&text, fmt) {
                return Ok(Moment {
                    utc: dt.with_timezone(&Utc),
                    zone: Zone::Fixed(*dt.offset()),
                });
            }
        }

        const NAIVE_FORMATS: [&str; 4] = [
            "%Y-%m-%dT%H:%M:%S%.f",
            "%Y-%m-%d %H:%M:%S%.f",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M",
        ];
        let naive = NAIVE_FORMATS
            .iter()
            .find_map(|fmt| NaiveDateTime::parse_from_str(&text, fmt).ok())
            .or_else(|| {
                NaiveDate::parse_from_str(&text, "%Y-%m-%d")
                    .ok()
                    .and_then(|d| d.and_hms_opt(0, 0, 0))
            })
            .ok_or_else(|| {
                format!("Could not parse datetime '{value}': Invalid isoformat string: '{value}'")
            })?;

        let zone = Zone::named(default_tz)?;
        let utc = zone.resolve_local(naive).ok_or_else(|| {
            format!(
                "Local datetime '{value}' does not exist in timezone '{default_tz}'. \
                 Choose an existing local time or include an explicit UTC offset."
            )
        })?;
        Ok(Moment { utc, zone })
    }

    /// Render a duration like '2 days, 3 hours, 15 minutes' (negative durations
    /// are prefixed with '-'). Sub-second parts are truncated.
    ///
    /// ```
    /// use atomic_forge::datetime_tool::DateTimeTool;
    /// assert_eq!(DateTimeTool::humanize_seconds(183_300), "2 days, 2 hours, 55 minutes");
    /// assert_eq!(DateTimeTool::humanize_seconds(-1), "-1 second");
    /// assert_eq!(DateTimeTool::humanize_seconds(0), "0 seconds");
    /// ```
    pub fn humanize_seconds(total: i64) -> String {
        let sign = if total < 0 { "-" } else { "" };
        let total = total.unsigned_abs();
        let (days, rem) = (total / 86_400, total % 86_400);
        let (hours, rem) = (rem / 3_600, rem % 3_600);
        let (minutes, seconds) = (rem / 60, rem % 60);
        let plural = |n: u64| if n == 1 { "" } else { "s" };
        let mut parts = Vec::new();
        if days > 0 {
            parts.push(format!("{days} day{}", plural(days)));
        }
        if hours > 0 {
            parts.push(format!("{hours} hour{}", plural(hours)));
        }
        if minutes > 0 {
            parts.push(format!("{minutes} minute{}", plural(minutes)));
        }
        if seconds > 0 || parts.is_empty() {
            parts.push(format!("{seconds} second{}", plural(seconds)));
        }
        format!("{sign}{}", parts.join(", "))
    }

    fn shift(m: Moment, input: &DateTimeToolInput) -> std::result::Result<Moment, String> {
        const RANGE: &str = "date value out of range";
        let mut utc = m.utc;
        // Calendar days keep the local clock time. A local time that lands in a
        // DST gap is normalised to its real instant.
        if input.days != 0 {
            let delta = Duration::try_days(input.days).ok_or(RANGE)?;
            let naive = m.local().checked_add_signed(delta).ok_or(RANGE)?;
            utc = m.zone.resolve_local_lenient(naive).ok_or(RANGE)?;
        }
        // Sub-day components are elapsed time, including across DST changes.
        let elapsed = Duration::try_hours(input.hours)
            .zip(Duration::try_minutes(input.minutes))
            .zip(Duration::try_seconds(input.seconds))
            .and_then(|((h, mi), s)| h.checked_add(&mi)?.checked_add(&s))
            .ok_or(RANGE)?;
        let utc = utc.checked_add_signed(elapsed).ok_or(RANGE)?;
        Ok(Moment { utc, zone: m.zone })
    }

    fn execute(input: &DateTimeToolInput) -> std::result::Result<DateTimeToolOutput, String> {
        let op = input.operation;
        match op {
            DateTimeOperation::Now => {
                let zone = Zone::named(&input.timezone)?;
                Ok(Moment { utc: Utc::now(), zone }.build("now"))
            }
            DateTimeOperation::Parse => {
                let value = input.input.as_deref().filter(|s| !s.is_empty()).ok_or("'parse' operation requires `input`.")?;
                Ok(Self::parse(value, &input.timezone)?.build("parse"))
            }
            DateTimeOperation::Convert => {
                let (value, target) = input
                    .input
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .zip(input.target_timezone.as_deref().filter(|s| !s.is_empty()))
                    .ok_or("'convert' operation requires `input` and `target_timezone`.")?;
                let m = Self::parse(value, &input.timezone)?;
                let zone = Zone::named(target)?;
                Ok(Moment { utc: m.utc, zone }.build("convert"))
            }
            DateTimeOperation::Shift => {
                let value = input.input.as_deref().filter(|s| !s.is_empty()).ok_or("'shift' operation requires `input`.")?;
                let m = Self::parse(value, &input.timezone)?;
                Ok(Self::shift(m, input)?.build("shift"))
            }
            DateTimeOperation::Diff => {
                let (a, b) = input
                    .input
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .zip(input.input_2.as_deref().filter(|s| !s.is_empty()))
                    .ok_or("'diff' operation requires `input` and `input_2`.")?;
                // Subtract instants, so a DST change between the two is counted.
                let a = Self::parse(a, &input.timezone)?;
                let b = Self::parse(b, &input.timezone)?;
                let delta = b.utc - a.utc;
                let secs = delta.num_seconds();
                let micros = delta.subsec_nanos() / 1_000;
                Ok(DateTimeToolOutput {
                    operation: "diff".to_string(),
                    diff_seconds: Some(secs as f64 + f64::from(micros) / 1e6),
                    diff_human: Some(Self::humanize_seconds(secs)),
                    ..Default::default()
                })
            }
        }
    }
}

impl Tool for DateTimeTool {
    type Input = DateTimeToolInput;
    type Output = DateTimeToolOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        Ok(Self::execute(&input).unwrap_or_else(|error| DateTimeToolOutput {
            operation: input.operation.as_str().to_string(),
            error: Some(error),
            ..Default::default()
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> DateTimeTool {
        DateTimeTool::new(DateTimeToolConfig::default())
    }

    fn input(op: DateTimeOperation, tz: &str, value: &str) -> DateTimeToolInput {
        let mut i = DateTimeToolInput::new(op);
        i.timezone = tz.to_string();
        i.input = Some(value.to_string());
        i
    }

    #[test]
    fn schema_metadata() {
        assert!(DateTimeToolInput::schema_description().contains("date and time operations"));
        let schema = serde_json::to_value(DateTimeToolInput::generate_schema()).unwrap();
        assert!(schema["properties"].get("operation").is_some());
        let out = serde_json::to_value(DateTimeToolOutput::generate_schema()).unwrap();
        assert!(out["properties"].get("diff_human").is_some());
    }

    #[test]
    fn input_defaults_and_enum() {
        let i: DateTimeToolInput = serde_json::from_str(r#"{"operation":"now"}"#).unwrap();
        assert_eq!(i.timezone, "UTC");
        assert_eq!(i.days, 0);
        assert!(serde_json::from_str::<DateTimeToolInput>(r#"{"operation":"bogus"}"#).is_err());
    }

    #[test]
    fn now_has_fields() {
        let mut i = DateTimeToolInput::new(DateTimeOperation::Now);
        i.timezone = "Asia/Tokyo".into();
        let out = tool().run(i).unwrap();
        assert!(out.error.is_none());
        assert_eq!(out.timezone.as_deref(), Some("Asia/Tokyo"));
        assert!(out.iso.unwrap().ends_with("+09:00"));
        assert!(out.human.unwrap().ends_with("JST"));
    }

    #[test]
    fn parse_naive_in_zone() {
        let out = tool()
            .run(input(DateTimeOperation::Parse, "Europe/Brussels", "2026-04-29T15:30:00"))
            .unwrap();
        assert_eq!(out.iso.as_deref(), Some("2026-04-29T15:30:00+02:00"));
        assert_eq!(out.weekday.as_deref(), Some("Wednesday"));
        assert_eq!(out.human.as_deref(), Some("Wed, 29 Apr 2026 15:30 CEST"));
        assert_eq!(out.unix_timestamp, Some(1_777_469_400.0));
        assert_eq!((out.year, out.month, out.day), (Some(2026), Some(4), Some(29)));
    }

    #[test]
    fn parse_z_and_offset_and_date_only() {
        let t = tool();
        let z = t.run(input(DateTimeOperation::Parse, "UTC", "2026-04-29T15:30:00Z")).unwrap();
        assert_eq!(z.iso.as_deref(), Some("2026-04-29T15:30:00+00:00"));
        assert_eq!(z.timezone.as_deref(), Some("UTC"));
        let off = t
            .run(input(DateTimeOperation::Parse, "UTC", "2026-04-29T15:30:00+05:30"))
            .unwrap();
        assert_eq!(off.iso.as_deref(), Some("2026-04-29T15:30:00+05:30"));
        assert_eq!(off.timezone.as_deref(), Some("UTC+05:30"));
        let d = t.run(input(DateTimeOperation::Parse, "UTC", "2026-04-29")).unwrap();
        assert_eq!(d.iso.as_deref(), Some("2026-04-29T00:00:00+00:00"));
        let frac = t
            .run(input(DateTimeOperation::Parse, "UTC", "2026-04-29 15:30:00.25"))
            .unwrap();
        assert_eq!(frac.iso.as_deref(), Some("2026-04-29T15:30:00.250000+00:00"));
    }

    #[test]
    fn parse_errors() {
        let t = tool();
        let bad = t.run(input(DateTimeOperation::Parse, "UTC", "not a date")).unwrap();
        assert!(bad.error.unwrap().contains("Could not parse datetime"));
        assert_eq!(bad.operation, "parse");
        let tz = t.run(input(DateTimeOperation::Parse, "Mars/Base", "2026-01-01")).unwrap();
        assert!(tz.error.unwrap().contains("Unknown timezone 'Mars/Base'"));
        let missing = t.run(DateTimeToolInput::new(DateTimeOperation::Parse)).unwrap();
        assert!(missing.error.unwrap().contains("requires `input`"));
    }

    #[test]
    fn parse_dst_gap_is_error_and_overlap_is_first() {
        let t = tool();
        let gap = t
            .run(input(DateTimeOperation::Parse, "America/New_York", "2026-03-08T02:30:00"))
            .unwrap();
        assert!(gap.error.unwrap().contains("does not exist"));
        let overlap = t
            .run(input(DateTimeOperation::Parse, "America/New_York", "2026-11-01T01:30:00"))
            .unwrap();
        assert_eq!(overlap.iso.as_deref(), Some("2026-11-01T01:30:00-04:00"));
    }

    #[test]
    fn convert_between_zones() {
        let mut i = input(DateTimeOperation::Convert, "Europe/Brussels", "2026-04-29T15:30:00");
        i.target_timezone = Some("America/New_York".into());
        let out = tool().run(i).unwrap();
        assert_eq!(out.iso.as_deref(), Some("2026-04-29T09:30:00-04:00"));
        assert_eq!(out.timezone.as_deref(), Some("America/New_York"));
        assert_eq!(out.human.as_deref(), Some("Wed, 29 Apr 2026 09:30 EDT"));

        let missing = tool()
            .run(input(DateTimeOperation::Convert, "UTC", "2026-04-29T15:30:00"))
            .unwrap();
        assert!(missing.error.unwrap().contains("target_timezone"));
    }

    #[test]
    fn shift_days_and_hours() {
        let mut i = input(DateTimeOperation::Shift, "UTC", "2026-04-29T15:30:00");
        i.days = 3;
        i.hours = -2;
        let out = tool().run(i).unwrap();
        assert_eq!(out.iso.as_deref(), Some("2026-05-02T13:30:00+00:00"));
    }

    #[test]
    fn shift_subday_is_elapsed_time_across_dst() {
        // New York springs forward at 02:00 on 2026-03-08.
        let mut hours = input(DateTimeOperation::Shift, "America/New_York", "2026-03-07T12:00:00");
        hours.hours = 24;
        let out = tool().run(hours).unwrap();
        assert_eq!(out.iso.as_deref(), Some("2026-03-08T13:00:00-04:00"));

        let mut days = input(DateTimeOperation::Shift, "America/New_York", "2026-03-07T12:00:00");
        days.days = 1;
        let out = tool().run(days).unwrap();
        assert_eq!(out.iso.as_deref(), Some("2026-03-08T12:00:00-04:00"));

        let mut mins = input(DateTimeOperation::Shift, "America/New_York", "2026-03-08T01:30:00");
        mins.minutes = 60;
        let out = tool().run(mins).unwrap();
        assert_eq!(out.iso.as_deref(), Some("2026-03-08T03:30:00-04:00"));

        let mut secs = input(DateTimeOperation::Shift, "America/New_York", "2026-11-01T00:30:00");
        secs.seconds = 3600 * 2;
        let out = tool().run(secs).unwrap();
        // 00:30 EDT + 2h elapsed = 01:30 EST (clocks fell back once).
        assert_eq!(out.iso.as_deref(), Some("2026-11-01T01:30:00-05:00"));
    }

    #[test]
    fn shift_calendar_day_into_gap_is_normalised() {
        let mut i = input(DateTimeOperation::Shift, "America/New_York", "2026-03-07T02:30:00");
        i.days = 1;
        let out = tool().run(i).unwrap();
        assert_eq!(out.iso.as_deref(), Some("2026-03-08T03:30:00-04:00"));
    }

    #[test]
    fn shift_out_of_range_is_error() {
        let mut i = input(DateTimeOperation::Shift, "UTC", "2026-04-29T15:30:00");
        i.days = i64::MAX;
        let out = tool().run(i).unwrap();
        assert!(out.error.unwrap().contains("out of range"));
    }

    #[test]
    fn diff_counts_elapsed_time_across_dst() {
        let mut i = input(DateTimeOperation::Diff, "America/New_York", "2026-03-08T00:00:00");
        i.input_2 = Some("2026-03-09T00:00:00".into());
        let out = tool().run(i).unwrap();
        assert_eq!(out.diff_seconds, Some(82_800.0));
        assert_eq!(out.diff_human.as_deref(), Some("23 hours"));
        assert!(out.iso.is_none());
    }

    #[test]
    fn diff_example_and_negative() {
        let mut i = input(DateTimeOperation::Diff, "UTC", "2026-04-29T15:30:00Z");
        i.input_2 = Some("2026-05-02T18:45:00Z".into());
        let out = tool().run(i).unwrap();
        assert_eq!(out.diff_human.as_deref(), Some("3 days, 3 hours, 15 minutes"));
        let mut r = input(DateTimeOperation::Diff, "UTC", "2026-05-02T18:45:00Z");
        r.input_2 = Some("2026-04-29T15:30:00Z".into());
        let out = tool().run(r).unwrap();
        assert_eq!(out.diff_human.as_deref(), Some("-3 days, 3 hours, 15 minutes"));
        let missing = tool().run(input(DateTimeOperation::Diff, "UTC", "2026-04-29")).unwrap();
        assert!(missing.error.unwrap().contains("input_2"));
    }

    #[test]
    fn humanize_edge_cases() {
        assert_eq!(DateTimeTool::humanize_seconds(1), "1 second");
        assert_eq!(DateTimeTool::humanize_seconds(3600), "1 hour");
        assert_eq!(DateTimeTool::humanize_seconds(86_400 + 60), "1 day, 1 minute");
    }
}
