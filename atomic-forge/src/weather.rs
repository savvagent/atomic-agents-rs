use std::collections::BTreeMap;
use std::time::Duration;

use atomic_agents::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Translate a WMO weather interpretation code into a human-readable description.
///
/// See <https://open-meteo.com/en/docs#weathervariables>.
///
/// ```
/// use atomic_forge::weather::describe_weather_code;
/// assert_eq!(describe_weather_code(Some(0)).as_deref(), Some("Clear sky"));
/// assert_eq!(describe_weather_code(Some(1234)).as_deref(), Some("Code 1234"));
/// assert_eq!(describe_weather_code(None), None);
/// ```
pub fn describe_weather_code(code: Option<i64>) -> Option<String> {
    let code = code?;
    let text = match code {
        0 => "Clear sky",
        1 => "Mainly clear",
        2 => "Partly cloudy",
        3 => "Overcast",
        45 => "Fog",
        48 => "Depositing rime fog",
        51 => "Light drizzle",
        53 => "Moderate drizzle",
        55 => "Dense drizzle",
        56 => "Light freezing drizzle",
        57 => "Dense freezing drizzle",
        61 => "Slight rain",
        63 => "Moderate rain",
        65 => "Heavy rain",
        66 => "Light freezing rain",
        67 => "Heavy freezing rain",
        71 => "Slight snow",
        73 => "Moderate snow",
        75 => "Heavy snow",
        77 => "Snow grains",
        80 => "Slight rain showers",
        81 => "Moderate rain showers",
        82 => "Violent rain showers",
        85 => "Slight snow showers",
        86 => "Heavy snow showers",
        95 => "Thunderstorm",
        96 => "Thunderstorm with slight hail",
        99 => "Thunderstorm with heavy hail",
        other => return Some(format!("Code {other}")),
    };
    Some(text.to_string())
}

/// Unit system for weather values.
#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WeatherUnits {
    /// Celsius, km/h and millimetres.
    #[default]
    Metric,
    /// Fahrenheit, mph and inches.
    Imperial,
}

fn default_language() -> String {
    "en".to_string()
}

fn default_forecast_days() -> u32 {
    3
}

/// Get current conditions and a daily/hourly forecast for a location. The
/// location can be a city name ('Brussels', 'New York', 'San Francisco, CA')
/// or a latitude/longitude pair as 'lat,lon'. Powered by the free Open-Meteo
/// API - no key required.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct WeatherInput {
    /// City name (e.g. 'Tokyo', 'Brussels'), city + admin ('Springfield, IL'),
    /// or a `lat,lon` pair (e.g. '48.8566,2.3522').
    pub location: String,
    /// Optional ISO 3166-1 alpha-2 country filter for city lookup (e.g. GB).
    /// Ignored for lat,lon input.
    #[serde(default)]
    pub country_code: Option<String>,
    /// 'metric' uses Celsius and km/h; 'imperial' uses Fahrenheit and mph.
    #[serde(default)]
    pub units: WeatherUnits,
    /// Number of forecast days to return (1-16).
    #[serde(default = "default_forecast_days")]
    pub forecast_days: u32,
    /// Whether to include the hourly forecast.
    #[serde(default)]
    pub include_hourly: bool,
    /// Geocoder language hint (ISO 639-1).
    #[serde(default = "default_language")]
    pub language: String,
}

/// Current observed weather.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq)]
pub struct WeatherCurrent {
    /// Local timestamp of the observation (ISO 8601).
    pub time: String,
    /// Temperature in the requested unit, or null when unavailable.
    pub temperature: Option<f64>,
    /// 'Feels like' temperature.
    pub apparent_temperature: Option<f64>,
    /// Relative humidity, in percent.
    pub humidity: Option<i64>,
    /// Wind speed in km/h or mph (matches `units`).
    pub wind_speed: Option<f64>,
    /// Wind direction in degrees (0-360).
    pub wind_direction: Option<i64>,
    /// Precipitation amount, in mm (metric) or inches (imperial).
    pub precipitation: Option<f64>,
    /// WMO weather code.
    pub weather_code: Option<i64>,
    /// Human-readable description of the weather code.
    pub weather_description: Option<String>,
    /// True when the sun is up at the location.
    pub is_day: Option<bool>,
}

/// Daily forecast bucket.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq)]
pub struct WeatherDay {
    /// Local calendar date (YYYY-MM-DD).
    pub date: String,
    /// Daily maximum temperature.
    pub temperature_max: Option<f64>,
    /// Daily minimum temperature.
    pub temperature_min: Option<f64>,
    /// Daily 'feels like' max.
    pub apparent_temperature_max: Option<f64>,
    /// Daily 'feels like' min.
    pub apparent_temperature_min: Option<f64>,
    /// Total precipitation for the day.
    pub precipitation_sum: Option<f64>,
    /// Max probability of precipitation, percent.
    pub precipitation_probability_max: Option<i64>,
    /// WMO weather code (representative for the day).
    pub weather_code: Option<i64>,
    /// Human-readable description of the day's weather code.
    pub weather_description: Option<String>,
    /// Local sunrise time (ISO 8601).
    pub sunrise: Option<String>,
    /// Local sunset time (ISO 8601).
    pub sunset: Option<String>,
    /// Maximum wind speed.
    pub wind_speed_max: Option<f64>,
}

/// Hourly forecast point (only populated when include_hourly is true).
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug, PartialEq)]
pub struct WeatherHour {
    /// Local timestamp (ISO 8601).
    pub time: String,
    /// Temperature.
    pub temperature: Option<f64>,
    /// Probability of precipitation, percent.
    pub precipitation_probability: Option<i64>,
    /// WMO weather code.
    pub weather_code: Option<i64>,
    /// Human-readable description of the weather code.
    pub weather_description: Option<String>,
}

/// Output of the weather tool.
#[derive(IOSchema, Serialize, Deserialize, JsonSchema, Clone, Debug)]
pub struct WeatherOutput {
    /// Resolved location label.
    pub location_name: String,
    /// Country of the resolved location.
    pub country: Option<String>,
    /// Latitude in degrees.
    pub latitude: f64,
    /// Longitude in degrees.
    pub longitude: f64,
    /// IANA timezone of the location.
    pub timezone: Option<String>,
    /// Units used for the response, keyed by variable name, e.g.
    /// {'temperature': '°C', 'wind_speed': 'km/h', 'precipitation': 'mm'}.
    pub units: BTreeMap<String, String>,
    /// Current conditions.
    pub current: Option<WeatherCurrent>,
    /// Daily forecast buckets.
    #[serde(default)]
    pub daily: Vec<WeatherDay>,
    /// Hourly forecast points (when requested).
    #[serde(default)]
    pub hourly: Vec<WeatherHour>,
    /// Error message when the operation failed.
    #[serde(default)]
    pub error: Option<String>,
}

/// Configuration for the weather tool.
pub struct WeatherConfig {
    pub component: ComponentConfig,
    /// Open-Meteo geocoding endpoint.
    pub geocoding_url: String,
    /// Open-Meteo forecast endpoint.
    pub forecast_url: String,
    /// User agent for HTTP requests.
    pub user_agent: String,
    /// HTTP request timeout.
    pub timeout: Duration,
}

impl Default for WeatherConfig {
    fn default() -> Self {
        Self {
            component: ComponentConfig::new()
                .with_title("Weather")
                .with_description("Current weather and forecast via Open-Meteo"),
            geocoding_url: "https://geocoding-api.open-meteo.com/v1/search".to_string(),
            forecast_url: "https://api.open-meteo.com/v1/forecast".to_string(),
            user_agent: "atomic-agents-weather-tool/1.0 (+https://github.com/savvagent/atomic-agents-rs)"
                .to_string(),
            timeout: Duration::from_secs(15),
        }
    }
}

/// Weather tool backed by the free Open-Meteo APIs (geocoding + forecast).
///
/// # Example
///
/// ```no_run
/// use atomic_agents::prelude::*;
/// use atomic_forge::weather::*;
///
/// # #[tokio::main(flavor = "multi_thread")]
/// # async fn main() {
/// let tool = WeatherTool::new(WeatherConfig::default());
/// let out = tool.run(serde_json::from_str(r#"{"location": "Brussels"}"#).unwrap()).unwrap();
/// println!("{:?}", out.current);
/// # }
/// ```
pub struct WeatherTool {
    config: ComponentConfig,
    geocoding_url: String,
    forecast_url: String,
    client: reqwest::Client,
}

/// A resolved geographic location.
#[derive(Clone, Debug, PartialEq)]
struct Place {
    name: String,
    country: Option<String>,
    latitude: f64,
    longitude: f64,
}

fn num_f64(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64)
}

fn num_i64(v: Option<&Value>) -> Option<i64> {
    v.and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f.round() as i64)))
}

/// Value at `index` of an array stored under `key`.
fn at<'a>(obj: &'a Value, key: &str, index: usize) -> Option<&'a Value> {
    obj.get(key)?.as_array()?.get(index).filter(|v| !v.is_null())
}

impl WeatherTool {
    pub fn new(config: WeatherConfig) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(config.user_agent)
            .timeout(config.timeout)
            .build()
            .unwrap_or_default();
        Self {
            config: config.component,
            geocoding_url: config.geocoding_url,
            forecast_url: config.forecast_url,
            client,
        }
    }

    /// Parse a `lat,lon` pair, returning `None` when malformed or out of range.
    ///
    /// ```
    /// use atomic_forge::weather::WeatherTool;
    /// assert_eq!(WeatherTool::parse_latlon("48.8566, 2.3522"), Some((48.8566, 2.3522)));
    /// assert_eq!(WeatherTool::parse_latlon("91,0"), None);
    /// assert_eq!(WeatherTool::parse_latlon("Paris"), None);
    /// ```
    pub fn parse_latlon(value: &str) -> Option<(f64, f64)> {
        let (a, b) = value.split_once(',')?;
        let number = |s: &str| -> Option<f64> {
            let s = s.trim();
            let digits = s.strip_prefix('-').unwrap_or(s);
            let valid = !digits.is_empty()
                && digits.split('.').count() <= 2
                && digits.split('.').all(|p| p.bytes().all(|b| b.is_ascii_digit()))
                && !digits.starts_with('.')
                && !digits.ends_with('.');
            if valid { s.parse().ok() } else { None }
        };
        let (lat, lon) = (number(a)?, number(b)?);
        ((-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon)).then_some((lat, lon))
    }

    async fn get_json(
        &self,
        url: &str,
        params: &[(&str, String)],
        what: &str,
    ) -> std::result::Result<Value, String> {
        let response = self
            .client
            .get(url)
            .header("Accept", "application/json")
            .query(params)
            .send()
            .await
            .map_err(|e| format!("{what} request failed: {e}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("{what} request failed: {status}"));
        }
        response
            .json()
            .await
            .map_err(|e| format!("Failed to parse {what} response: {e}"))
    }

    async fn geocode(
        &self,
        query: &str,
        language: &str,
        country_code: Option<&str>,
    ) -> std::result::Result<Place, String> {
        let mut params = vec![
            ("name", query.to_string()),
            ("count", "1".to_string()),
            ("language", language.to_string()),
            ("format", "json".to_string()),
        ];
        if let Some(cc) = country_code {
            params.push(("countryCode", cc.to_ascii_uppercase()));
        }
        let data = self.get_json(&self.geocoding_url, &params, "geocoding").await?;
        Self::parse_geocode(&data, query)
    }

    fn parse_geocode(data: &Value, query: &str) -> std::result::Result<Place, String> {
        let first = data
            .get("results")
            .and_then(Value::as_array)
            .and_then(|r| r.first())
            .ok_or_else(|| format!("Location '{query}' not found."))?;
        let latitude = num_f64(first.get("latitude"))
            .ok_or_else(|| "Geocoding result is missing latitude".to_string())?;
        let longitude = num_f64(first.get("longitude"))
            .ok_or_else(|| "Geocoding result is missing longitude".to_string())?;
        let mut name = first
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(query)
            .to_string();
        if let Some(admin) = first
            .get("admin1")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            name = format!("{name}, {admin}");
        }
        Ok(Place {
            name,
            country: first.get("country").and_then(Value::as_str).map(String::from),
            latitude,
            longitude,
        })
    }

    async fn forecast(&self, place: &Place, input: &WeatherInput) -> std::result::Result<Value, String> {
        let metric = input.units == WeatherUnits::Metric;
        let mut params = vec![
            ("latitude", place.latitude.to_string()),
            ("longitude", place.longitude.to_string()),
            ("timezone", "auto".to_string()),
            ("forecast_days", input.forecast_days.clamp(1, 16).to_string()),
            ("temperature_unit", if metric { "celsius" } else { "fahrenheit" }.to_string()),
            ("wind_speed_unit", if metric { "kmh" } else { "mph" }.to_string()),
            ("precipitation_unit", if metric { "mm" } else { "inch" }.to_string()),
            (
                "current",
                "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,\
                 wind_direction_10m,precipitation,weather_code,is_day"
                    .to_string(),
            ),
            (
                "daily",
                "temperature_2m_max,temperature_2m_min,apparent_temperature_max,\
                 apparent_temperature_min,precipitation_sum,precipitation_probability_max,\
                 weather_code,sunrise,sunset,wind_speed_10m_max"
                    .to_string(),
            ),
        ];
        if input.include_hourly {
            params.push((
                "hourly",
                "temperature_2m,precipitation_probability,weather_code".to_string(),
            ));
        }
        self.get_json(&self.forecast_url, &params, "forecast").await
    }

    /// Build the `current` block from a forecast response.
    pub fn build_current(data: &Value) -> Option<WeatherCurrent> {
        let c = data.get("current").filter(|c| c.is_object() && !c.as_object().unwrap().is_empty())?;
        let code = num_i64(c.get("weather_code"));
        Some(WeatherCurrent {
            time: c.get("time").and_then(Value::as_str).unwrap_or("").to_string(),
            temperature: num_f64(c.get("temperature_2m")),
            apparent_temperature: num_f64(c.get("apparent_temperature")),
            humidity: num_i64(c.get("relative_humidity_2m")),
            wind_speed: num_f64(c.get("wind_speed_10m")),
            wind_direction: num_i64(c.get("wind_direction_10m")),
            precipitation: num_f64(c.get("precipitation")),
            weather_code: code,
            weather_description: describe_weather_code(code),
            is_day: num_i64(c.get("is_day")).map(|v| v != 0),
        })
    }

    /// Build the daily forecast buckets from a forecast response.
    pub fn build_daily(data: &Value) -> Vec<WeatherDay> {
        let Some(daily) = data.get("daily") else {
            return Vec::new();
        };
        let dates = daily.get("time").and_then(Value::as_array).cloned().unwrap_or_default();
        dates
            .iter()
            .enumerate()
            .map(|(i, date)| {
                let code = num_i64(at(daily, "weather_code", i));
                let text = |k: &str| at(daily, k, i).and_then(Value::as_str).map(String::from);
                WeatherDay {
                    date: date.as_str().unwrap_or("").to_string(),
                    temperature_max: num_f64(at(daily, "temperature_2m_max", i)),
                    temperature_min: num_f64(at(daily, "temperature_2m_min", i)),
                    apparent_temperature_max: num_f64(at(daily, "apparent_temperature_max", i)),
                    apparent_temperature_min: num_f64(at(daily, "apparent_temperature_min", i)),
                    precipitation_sum: num_f64(at(daily, "precipitation_sum", i)),
                    precipitation_probability_max: num_i64(at(daily, "precipitation_probability_max", i)),
                    weather_code: code,
                    weather_description: describe_weather_code(code),
                    sunrise: text("sunrise"),
                    sunset: text("sunset"),
                    wind_speed_max: num_f64(at(daily, "wind_speed_10m_max", i)),
                }
            })
            .collect()
    }

    /// Build the hourly forecast points from a forecast response.
    pub fn build_hourly(data: &Value) -> Vec<WeatherHour> {
        let Some(hourly) = data.get("hourly") else {
            return Vec::new();
        };
        let times = hourly.get("time").and_then(Value::as_array).cloned().unwrap_or_default();
        times
            .iter()
            .enumerate()
            .map(|(i, time)| {
                let code = num_i64(at(hourly, "weather_code", i));
                WeatherHour {
                    time: time.as_str().unwrap_or("").to_string(),
                    temperature: num_f64(at(hourly, "temperature_2m", i)),
                    precipitation_probability: num_i64(at(hourly, "precipitation_probability", i)),
                    weather_code: code,
                    weather_description: describe_weather_code(code),
                }
            })
            .collect()
    }

    async fn fetch(&self, input: &WeatherInput) -> std::result::Result<WeatherOutput, String> {
        let place = match Self::parse_latlon(&input.location) {
            Some((latitude, longitude)) => Place {
                name: format!("{latitude:.4}, {longitude:.4}"),
                country: None,
                latitude,
                longitude,
            },
            None => {
                let cc = input.country_code.as_deref().filter(|s| !s.is_empty());
                if let Some(cc) = cc
                    && !(cc.len() == 2 && cc.bytes().all(|b| b.is_ascii_alphabetic()))
                {
                    return Err(format!(
                        "Invalid country_code '{cc}': expected an ISO 3166-1 alpha-2 code."
                    ));
                }
                self.geocode(&input.location, &input.language, cc).await?
            }
        };
        let data = self.forecast(&place, input).await?;
        let metric = input.units == WeatherUnits::Metric;
        let units = BTreeMap::from([
            ("temperature".to_string(), if metric { "°C" } else { "°F" }.to_string()),
            ("wind_speed".to_string(), if metric { "km/h" } else { "mph" }.to_string()),
            ("precipitation".to_string(), if metric { "mm" } else { "in" }.to_string()),
        ]);
        Ok(WeatherOutput {
            location_name: place.name,
            country: place.country,
            latitude: place.latitude,
            longitude: place.longitude,
            timezone: data.get("timezone").and_then(Value::as_str).map(String::from),
            units,
            current: Self::build_current(&data),
            daily: Self::build_daily(&data),
            hourly: if input.include_hourly {
                Self::build_hourly(&data)
            } else {
                Vec::new()
            },
            error: None,
        })
    }
}

impl Tool for WeatherTool {
    type Input = WeatherInput;
    type Output = WeatherOutput;

    fn config(&self) -> &ComponentConfig {
        &self.config
    }

    fn run(&self, input: Self::Input) -> Result<Self::Output> {
        let outcome = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.fetch(&input))
        });
        Ok(outcome.unwrap_or_else(|error| WeatherOutput {
            location_name: input.location.clone(),
            country: None,
            latitude: 0.0,
            longitude: 0.0,
            timezone: None,
            units: BTreeMap::new(),
            current: None,
            daily: Vec::new(),
            hourly: Vec::new(),
            error: Some(error),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn input(location: &str) -> WeatherInput {
        serde_json::from_value(json!({ "location": location })).unwrap()
    }

    fn tool_for(uri: &str) -> WeatherTool {
        WeatherTool::new(WeatherConfig {
            geocoding_url: format!("{uri}/geo"),
            forecast_url: format!("{uri}/forecast"),
            ..Default::default()
        })
    }

    fn forecast_body() -> Value {
        json!({
            "timezone": "Europe/Brussels",
            "current": {
                "time": "2026-04-29T15:00", "temperature_2m": 14.5, "apparent_temperature": 12.0,
                "relative_humidity_2m": 60, "wind_speed_10m": 11.2, "wind_direction_10m": 250,
                "precipitation": 0.0, "weather_code": 3, "is_day": 1
            },
            "daily": {
                "time": ["2026-04-29", "2026-04-30"],
                "temperature_2m_max": [16.0, 18.5], "temperature_2m_min": [7.0, null],
                "weather_code": [61, 999], "sunrise": ["2026-04-29T06:10", "2026-04-30T06:08"],
                "precipitation_probability_max": [80, 10]
            },
            "hourly": {
                "time": ["2026-04-29T00:00"], "temperature_2m": [9.0],
                "precipitation_probability": [5], "weather_code": [0]
            }
        })
    }

    #[test]
    fn schema_metadata() {
        assert!(WeatherInput::schema_description().contains("Open-Meteo"));
        let schema = serde_json::to_value(WeatherInput::generate_schema()).unwrap();
        assert!(schema["properties"].get("forecast_days").is_some());
        let out = serde_json::to_value(WeatherOutput::generate_schema()).unwrap();
        assert!(out["properties"].get("daily").is_some());
    }

    #[test]
    fn defaults_apply() {
        let i = input("Oslo");
        assert_eq!(i.units, WeatherUnits::Metric);
        assert_eq!(i.forecast_days, 3);
        assert_eq!(i.language, "en");
        assert!(!i.include_hourly);
    }

    #[test]
    fn latlon_parsing() {
        assert_eq!(WeatherTool::parse_latlon("48.8566,2.3522"), Some((48.8566, 2.3522)));
        assert_eq!(WeatherTool::parse_latlon(" -33.9 , 151.2 "), Some((-33.9, 151.2)));
        assert_eq!(WeatherTool::parse_latlon("91,0"), None);
        assert_eq!(WeatherTool::parse_latlon("0,181"), None);
        assert_eq!(WeatherTool::parse_latlon("Springfield, IL"), None);
        assert_eq!(WeatherTool::parse_latlon("1e3,2"), None);
        assert_eq!(WeatherTool::parse_latlon("1,2,3"), None);
    }

    #[test]
    fn weather_codes() {
        assert_eq!(describe_weather_code(Some(95)).as_deref(), Some("Thunderstorm"));
        assert_eq!(describe_weather_code(Some(42)).as_deref(), Some("Code 42"));
    }

    #[test]
    fn builders_parse_canned_forecast() {
        let data = forecast_body();
        let cur = WeatherTool::build_current(&data).unwrap();
        assert_eq!(cur.temperature, Some(14.5));
        assert_eq!(cur.humidity, Some(60));
        assert_eq!(cur.is_day, Some(true));
        assert_eq!(cur.weather_description.as_deref(), Some("Overcast"));
        let days = WeatherTool::build_daily(&data);
        assert_eq!(days.len(), 2);
        assert_eq!(days[0].weather_description.as_deref(), Some("Slight rain"));
        assert_eq!(days[1].temperature_min, None);
        assert_eq!(days[1].weather_description.as_deref(), Some("Code 999"));
        assert_eq!(days[1].sunrise.as_deref(), Some("2026-04-30T06:08"));
        let hours = WeatherTool::build_hourly(&data);
        assert_eq!(hours.len(), 1);
        assert_eq!(hours[0].weather_description.as_deref(), Some("Clear sky"));
        assert!(WeatherTool::build_current(&json!({})).is_none());
    }

    #[test]
    fn geocode_parsing() {
        let data = json!({"results": [{"name": "Springfield", "admin1": "Illinois",
            "country": "United States", "latitude": 39.8, "longitude": -89.6}]});
        let place = WeatherTool::parse_geocode(&data, "Springfield").unwrap();
        assert_eq!(place.name, "Springfield, Illinois");
        assert_eq!(place.country.as_deref(), Some("United States"));
        assert!(WeatherTool::parse_geocode(&json!({}), "Nowhere").unwrap_err().contains("not found"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn run_city_end_to_end() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/geo"))
            .and(query_param("name", "Brussels"))
            .and(query_param("countryCode", "BE"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{"name": "Brussels", "country": "Belgium", "latitude": 50.85, "longitude": 4.35}]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/forecast"))
            .and(query_param("temperature_unit", "fahrenheit"))
            .and(query_param("wind_speed_unit", "mph"))
            .respond_with(ResponseTemplate::new(200).set_body_json(forecast_body()))
            .mount(&server)
            .await;
        let tool = tool_for(&server.uri());
        let mut i = input("Brussels");
        i.country_code = Some("be".into());
        i.units = WeatherUnits::Imperial;
        i.include_hourly = true;
        let out = tool.run(i).unwrap();
        assert!(out.error.is_none(), "{:?}", out.error);
        assert_eq!(out.location_name, "Brussels");
        assert_eq!(out.country.as_deref(), Some("Belgium"));
        assert_eq!(out.timezone.as_deref(), Some("Europe/Brussels"));
        assert_eq!(out.units["temperature"], "°F");
        assert_eq!(out.daily.len(), 2);
        assert_eq!(out.hourly.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn run_latlon_skips_geocoding() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/forecast"))
            .and(query_param("latitude", "48.8566"))
            .respond_with(ResponseTemplate::new(200).set_body_json(forecast_body()))
            .mount(&server)
            .await;
        let tool = tool_for(&server.uri());
        let out = tool.run(input("48.8566,2.3522")).unwrap();
        assert!(out.error.is_none());
        assert_eq!(out.location_name, "48.8566, 2.3522");
        assert!(out.hourly.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unknown_location_sets_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/geo"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
            .mount(&server)
            .await;
        let tool = tool_for(&server.uri());
        let out = tool.run(input("Atlantis")).unwrap();
        assert!(out.error.unwrap().contains("not found"));
        assert!(out.current.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn http_error_sets_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let tool = tool_for(&server.uri());
        let out = tool.run(input("10,10")).unwrap();
        assert!(out.error.unwrap().contains("500"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn invalid_country_code_sets_error() {
        let tool = tool_for("http://127.0.0.1:1");
        let mut i = input("Paris");
        i.country_code = Some("FRA".into());
        let out = tool.run(i).unwrap();
        assert!(out.error.unwrap().contains("country_code"));
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires network access"]
    async fn live_weather() {
        let tool = WeatherTool::new(WeatherConfig::default());
        let out = tool.run(input("Brussels")).unwrap();
        assert!(out.error.is_none());
        assert!(out.current.is_some());
    }
}
