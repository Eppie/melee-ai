use pyo3::once_cell::GILOnceCell;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3::{Py, PyObject};
use std::collections::HashMap;
use crate::enums::{Action, Character, Menu, Stage, SubMenu, ProjectileType};

const BUTTON_NAMES: &[&str] = &[
    "BUTTON_A",
    "BUTTON_B",
    "BUTTON_X",
    "BUTTON_Y",
    "BUTTON_Z",
    "BUTTON_L",
    "BUTTON_R",
    "BUTTON_START",
    "BUTTON_D_UP",
    "BUTTON_D_DOWN",
    "BUTTON_D_LEFT",
    "BUTTON_D_RIGHT",
    "BUTTON_MAIN",
    "BUTTON_C",
];

static BUTTON_CLASS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

fn canonical_button_name(raw: &str) -> Option<&'static str> {
    let upper = raw.trim().to_ascii_uppercase();
    match upper.as_str() {
        "BUTTON_A" | "A" => Some("BUTTON_A"),
        "BUTTON_B" | "B" => Some("BUTTON_B"),
        "BUTTON_X" | "X" => Some("BUTTON_X"),
        "BUTTON_Y" | "Y" => Some("BUTTON_Y"),
        "BUTTON_Z" | "Z" => Some("BUTTON_Z"),
        "BUTTON_L" | "L" => Some("BUTTON_L"),
        "BUTTON_R" | "R" => Some("BUTTON_R"),
        "BUTTON_START" | "START" => Some("BUTTON_START"),
        "BUTTON_D_UP" | "D_UP" | "UP" => Some("BUTTON_D_UP"),
        "BUTTON_D_DOWN" | "D_DOWN" | "DOWN" => Some("BUTTON_D_DOWN"),
        "BUTTON_D_LEFT" | "D_LEFT" | "LEFT" => Some("BUTTON_D_LEFT"),
        "BUTTON_D_RIGHT" | "D_RIGHT" | "RIGHT" => Some("BUTTON_D_RIGHT"),
        "BUTTON_MAIN" | "MAIN" => Some("BUTTON_MAIN"),
        "BUTTON_C" | "C" => Some("BUTTON_C"),
        _ => None,
    }
}

fn button_class<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let class = BUTTON_CLASS.get_or_try_init(py, || {
        let module = py
            .import("melee.enums")
            .or_else(|_| py.import("libmelee.melee.enums"))?;
        let cls = module.getattr("Button")?;
        Ok(cls.into_py(py))
    })?;
    Ok(class.bind(py))
}

fn button_member(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let class = button_class(py)?;
    Ok(class.getattr(name)?.into_py(py))
}

fn init_button_dict(py: Python<'_>) -> Py<PyDict> {
    let dict = PyDict::new(py);
    let class = button_class(py).expect("Button enum not available");
    for &name in BUTTON_NAMES {
        let member = class
            .getattr(name)
            .unwrap_or_else(|_| panic!("Missing enum variant {}", name));
        dict.set_item(member, false)
            .unwrap_or_else(|_| panic!("Failed to initialize button {}", name));
    }
    dict.into()
}

fn name_from_py_object(py: Python<'_>, button: &PyAny) -> PyResult<String> {
    if let Ok(name_obj) = button.getattr("name") {
        let raw: String = name_obj.extract()?;
        if let Some(canonical) = canonical_button_name(&raw) {
            return Ok(canonical.to_string());
        }
    }

    if let Ok(value_obj) = button.getattr("value") {
        let raw: String = value_obj.extract()?;
        if let Some(canonical) = canonical_button_name(&raw) {
            return Ok(canonical.to_string());
        }
    }

    if let Ok(raw) = button.extract::<String>() {
        if let Some(canonical) = canonical_button_name(&raw) {
            return Ok(canonical.to_string());
        }
    }

    Err(pyo3::exceptions::PyValueError::new_err(
        "Unsupported button identifier",
    ))
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Position {
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
}

#[pymethods]
impl Position {
    #[new]
    fn new() -> Self {
        Position { x: 0.0, y: 0.0 }
    }
}

impl Default for Position {
    fn default() -> Self {
        Position { x: 0.0, y: 0.0 }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ECB {
    #[pyo3(get, set)]
    pub top: Position,
    #[pyo3(get, set)]
    pub bottom: Position,
    #[pyo3(get, set)]
    pub left: Position,
    #[pyo3(get, set)]
    pub right: Position,
}

#[pymethods]
impl ECB {
    #[new]
    fn new() -> Self {
        ECB {
            top: Position::default(),
            bottom: Position::default(),
            left: Position::default(),
            right: Position::default(),
        }
    }
}

impl Default for ECB {
    fn default() -> Self {
        ECB::new()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ControllerState {
    #[pyo3(get, set)]
    pub main_stick: (f32, f32),
    #[pyo3(get, set)]
    pub c_stick: (f32, f32),
    #[pyo3(get, set)]
    pub l_shoulder: f32,
    #[pyo3(get, set)]
    pub r_shoulder: f32,
    #[pyo3(get, set)]
    pub raw_main_stick: (i8, i8),
    buttons: Py<PyDict>,
}

#[pymethods]
impl ControllerState {
    #[new]
    fn new() -> Self {
        Python::with_gil(|py| ControllerState {
            main_stick: (0.5, 0.5),
            c_stick: (0.5, 0.5),
            l_shoulder: 0.0,
            r_shoulder: 0.0,
            raw_main_stick: (0, 0),
            buttons: init_button_dict(py),
        })
    }

    #[getter]
    fn button<'py>(&self, py: Python<'py>) -> &'py PyDict {
        self.buttons.bind(py)
    }

    pub fn set_button(&mut self, py: Python<'_>, button: &PyAny, pressed: bool) -> PyResult<()> {
        let name = name_from_py_object(py, button)?;
        self.set_button_flag(&name, pressed)
    }

    pub fn get_button(&self, py: Python<'_>, button: &PyAny) -> PyResult<bool> {
        let name = name_from_py_object(py, button)?;
        self.get_button_flag(&name)
    }
}

impl Default for ControllerState {
    fn default() -> Self {
        ControllerState::new()
    }
}

impl ControllerState {
    pub fn set_button_flag(&mut self, name: &str, pressed: bool) -> PyResult<()> {
        let Some(canonical) = canonical_button_name(name) else {
            return Err(pyo3::exceptions::PyValueError::new_err("Unknown button"));
        };
        Python::with_gil(|py| -> PyResult<()> {
            let dict = self.buttons.bind(py);
            let button_obj = button_member(py, canonical)?;
            dict.set_item(button_obj, pressed)?;
            Ok(())
        })
    }

    pub fn get_button_flag(&self, name: &str) -> PyResult<bool> {
        let Some(canonical) = canonical_button_name(name) else {
            return Ok(false);
        };
        Python::with_gil(|py| -> PyResult<bool> {
            let dict = self.buttons.bind(py);
            let button_obj = button_member(py, canonical)?;
            if let Some(value) = dict.get_item(button_obj) {
                value.extract::<bool>()
            } else {
                Ok(false)
            }
        })
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PlayerState {
    #[pyo3(get, set)]
    pub character: Character,
    #[pyo3(get, set)]
    pub character_selected: Character,
    #[pyo3(get, set)]
    pub position: Position,
    #[pyo3(get, set)]
    pub percent: i32,
    #[pyo3(get, set)]
    pub shield_strength: f32,
    #[pyo3(get, set)]
    pub is_powershield: bool,
    #[pyo3(get, set)]
    pub stock: u8,
    #[pyo3(get, set)]
    pub facing: bool,
    #[pyo3(get, set)]
    pub action: Action,
    #[pyo3(get, set)]
    pub action_frame: i32,
    #[pyo3(get, set)]
    pub invulnerable: bool,
    #[pyo3(get, set)]
    pub invulnerability_left: i32,
    #[pyo3(get, set)]
    pub hitlag_left: i32,
    #[pyo3(get, set)]
    pub hitstun_frames_left: i32,
    #[pyo3(get, set)]
    pub jumps_left: u8,
    #[pyo3(get, set)]
    pub on_ground: bool,
    #[pyo3(get, set)]
    pub speed_air_x_self: f32,
    #[pyo3(get, set)]
    pub speed_y_self: f32,
    #[pyo3(get, set)]
    pub speed_x_attack: f32,
    #[pyo3(get, set)]
    pub speed_y_attack: f32,
    #[pyo3(get, set)]
    pub speed_ground_x_self: f32,
    #[pyo3(get, set)]
    pub off_stage: bool,
    #[pyo3(get, set)]
    pub iasa: i32,
    #[pyo3(get, set)]
    pub controller_state: ControllerState,
    #[pyo3(get, set)]
    pub ecb: ECB,
    #[pyo3(get, set)]
    pub ecb_top: (f32, f32),
    #[pyo3(get, set)]
    pub ecb_bottom: (f32, f32),
    #[pyo3(get, set)]
    pub ecb_left: (f32, f32),
    #[pyo3(get, set)]
    pub ecb_right: (f32, f32),
    #[pyo3(get, set)]
    pub is_reflect_active: bool,
    #[pyo3(get, set)]
    pub is_subaction_invulnerable: bool,
    #[pyo3(get, set)]
    pub is_fastfalling: bool,
    #[pyo3(get, set)]
    pub is_defender_in_hitlag: bool,
    #[pyo3(get, set)]
    pub is_in_hitlag: bool,
    #[pyo3(get, set)]
    pub is_holding_character: bool,
    #[pyo3(get, set)]
    pub is_shield_active: bool,
    #[pyo3(get, set)]
    pub is_in_hitstun: bool,
    #[pyo3(get, set)]
    pub is_dead: bool,
    #[pyo3(get, set)]
    pub is_offscreen: bool,
    #[pyo3(get, set)]
    pub l_cancel_status: u8,
    #[pyo3(get, set)]
    pub cursor: Position,
    #[pyo3(get, set)]
    pub cursor_x: f32,
    #[pyo3(get, set)]
    pub cursor_y: f32,
    #[pyo3(get, set)]
    pub coin_down: bool,
    #[pyo3(get, set)]
    pub is_holding_cpu_slider: bool,
    #[pyo3(get, set)]
    pub cpu_level: u8,
    #[pyo3(get, set)]
    pub nickName: String,
    #[pyo3(get, set)]
    pub connectCode: String,
    #[pyo3(get, set)]
    pub displayName: String,
    #[pyo3(get, set)]
    pub team_id: u8,
    // Nana is handled differently - not exposed directly to Python
    nana: Option<Box<PlayerState>>,
}

#[pymethods]
impl PlayerState {
    #[new]
    pub fn new() -> Self {
        PlayerState {
            character: Character::default(),
            character_selected: Character::default(),
            position: Position::default(),
            percent: 0,
            shield_strength: 60.0,
            is_powershield: false,
            stock: 0,
            facing: true,
            action: Action::default(),
            action_frame: 0,
            invulnerable: false,
            invulnerability_left: 0,
            hitlag_left: 0,
            hitstun_frames_left: 0,
            jumps_left: 0,
            on_ground: true,
            speed_air_x_self: 0.0,
            speed_y_self: 0.0,
            speed_x_attack: 0.0,
            speed_y_attack: 0.0,
            speed_ground_x_self: 0.0,
            off_stage: false,
            iasa: 0,
            controller_state: ControllerState::default(),
            ecb: ECB::default(),
            ecb_top: (0.0, 0.0),
            ecb_bottom: (0.0, 0.0),
            ecb_left: (0.0, 0.0),
            ecb_right: (0.0, 0.0),
            is_reflect_active: false,
            is_subaction_invulnerable: false,
            is_fastfalling: false,
            is_defender_in_hitlag: false,
            is_in_hitlag: false,
            is_holding_character: false,
            is_shield_active: false,
            is_in_hitstun: false,
            is_dead: false,
            is_offscreen: false,
            l_cancel_status: 0,
            cursor: Position::default(),
            cursor_x: 0.0,
            cursor_y: 0.0,
            coin_down: false,
            is_holding_cpu_slider: false,
            cpu_level: 0,
            nickName: String::new(),
            connectCode: String::new(),
            displayName: String::new(),
            team_id: 0,
            nana: None,
        }
    }
    
    pub fn get_nana(&self) -> Option<PlayerState> {
        self.nana.as_ref().map(|n| (**n).clone())
    }
    
    pub fn set_nana(&mut self, nana: PlayerState) {
        self.nana = Some(Box::new(nana));
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Projectile {
    #[pyo3(get, set)]
    pub position: Position,
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
    #[pyo3(get, set)]
    pub speed: Position,
    #[pyo3(get, set)]
    pub x_speed: f32,
    #[pyo3(get, set)]
    pub y_speed: f32,
    #[pyo3(get, set)]
    pub owner: i32,
    #[pyo3(get, set)]
    pub proj_type: ProjectileType,
    #[pyo3(get, set)]
    pub frame: i32,
    #[pyo3(get, set)]
    pub subtype: u8,
}

#[pymethods]
impl Projectile {
    #[new]
    pub fn new() -> Self {
        Projectile {
            position: Position::default(),
            x: 0.0,
            y: 0.0,
            speed: Position::default(),
            x_speed: 0.0,
            y_speed: 0.0,
            owner: -1,
            proj_type: ProjectileType::default(),
            frame: 0,
            subtype: 0,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct GameState {
    #[pyo3(get, set)]
    pub frame: i32,
    #[pyo3(get, set)]
    pub stage: Stage,
    #[pyo3(get, set)]
    pub menu_state: Menu,
    #[pyo3(get, set)]
    pub submenu: SubMenu,
    // Store players as a HashMap to match Python dict interface
    players: HashMap<u8, PlayerState>,
    #[pyo3(get, set)]
    pub projectiles: Vec<Projectile>,
    #[pyo3(get, set)]
    pub ready_to_start: bool,
    #[pyo3(get, set)]
    pub is_teams: bool,
    #[pyo3(get, set)]
    pub distance: f32,
    #[pyo3(get, set)]
    pub menu_selection: i32,
    #[pyo3(get, set)]
    pub stage_select_cursor_x: f32,
    #[pyo3(get, set)]
    pub stage_select_cursor_y: f32,
    #[pyo3(get, set)]
    pub playedOn: String,
    #[pyo3(get, set)]
    pub startAt: String,
    #[pyo3(get, set)]
    pub consoleNick: String,
}

#[pymethods]
impl GameState {
    #[new]
    fn new() -> Self {
        GameState {
            frame: -10000,
            stage: Stage::default(),
            menu_state: Menu::default(),
            submenu: SubMenu::default(),
            players: HashMap::new(),
            projectiles: Vec::new(),
            ready_to_start: false,
            is_teams: false,
            distance: 0.0,
            menu_selection: 0,
            stage_select_cursor_x: 0.0,
            stage_select_cursor_y: 0.0,
            playedOn: String::new(),
            startAt: String::new(),
            consoleNick: String::new(),
        }
    }
    
    #[getter]
    pub fn players(&self) -> HashMap<u8, PlayerState> {
        self.players.clone()
    }
    
    fn get_player(&self, port: u8) -> Option<PlayerState> {
        self.players.get(&port).cloned()
    }
    
    fn set_player(&mut self, port: u8, player: PlayerState) {
        self.players.insert(port, player);
    }
    
    fn has_player(&self, port: u8) -> bool {
        self.players.contains_key(&port)
    }
    
    // Python dict-like interface for players
    fn __getitem__(&self, port: u8) -> PyResult<PlayerState> {
        self.players.get(&port)
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("Port {} not found", port)))
    }
    
    fn __setitem__(&mut self, port: u8, player: PlayerState) {
        self.players.insert(port, player);
    }
    
    fn __contains__(&self, port: u8) -> bool {
        self.players.contains_key(&port)
    }
    
    fn keys(&self) -> Vec<u8> {
        self.players.keys().copied().collect()
    }
    
    fn values(&self) -> Vec<PlayerState> {
        self.players.values().cloned().collect()
    }
    
    fn items(&self) -> Vec<(u8, PlayerState)> {
        self.players.iter().map(|(k, v)| (*k, v.clone())).collect()
    }
}

impl PlayerState {
    pub fn has_nana(&self) -> bool {
        self.nana.is_some()
    }
    
    pub fn ensure_nana(&mut self) -> &mut PlayerState {
        if self.nana.is_none() {
            self.nana = Some(Box::new(PlayerState::new()));
        }
        self.nana.as_mut().unwrap().as_mut()
    }
}

impl GameState {
    pub fn get_player_mut(&mut self, port: u8) -> Option<&mut PlayerState> {
        self.players.get_mut(&port)
    }
    
    pub fn ensure_player(&mut self, port: u8) -> &mut PlayerState {
        self.players.entry(port).or_insert_with(PlayerState::new)
    }
}
