use pyo3::prelude::*;
use pyo3::types::PyString;

// =====================
// Stage
// =====================

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Stage {
    NoStage = 0x00,
    FinalDestination = 0x19,
    Battlefield = 0x18,
    PokemonStadium = 0x12,
    Dreamland = 0x1A,
    FountainOfDreams = 0x08,
    YoshisStory = 0x06,
    RandomStage = 0x1D, // pseudo-stage, kept for parity with Python
}

#[pymethods]
impl Stage {
    #[getter]
    fn value(&self) -> u8 {
        *self as u8
    }

    /// Slippi/Dolphin "external" stage code -> internal Stage enum used here.
    #[staticmethod]
    pub fn to_internal_stage(stage_id: u16) -> Self {
        match stage_id {
            0x03 => Stage::PokemonStadium,
            0x08 => Stage::YoshisStory,
            0x02 => Stage::FountainOfDreams,
            0x1F => Stage::Battlefield,
            0x20 => Stage::FinalDestination,
            0x1C => Stage::Dreamland,
            _ => Stage::NoStage,
        }
    }
}

impl Stage {
    /// Direct map from the internal numeric stage IDs used alongside this enum.
    pub fn from_u16(val: u16) -> Self {
        match val {
            0x06 => Stage::YoshisStory,
            0x08 => Stage::FountainOfDreams,
            0x12 => Stage::PokemonStadium,
            0x18 => Stage::Battlefield,
            0x19 => Stage::FinalDestination,
            0x1A => Stage::Dreamland,
            0x1D => Stage::RandomStage,
            _ => Stage::NoStage,
        }
    }
}

impl Default for Stage {
    fn default() -> Self {
        Stage::NoStage
    }
}

// =====================
// Menu / SubMenu
// =====================

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Menu {
    CharacterSelect = 0,
    StageSelect = 1,
    InGame = 2,
    SuddenDeath = 3,
    PostgameScores = 4,
    MainMenu = 5,
    SlippiOnlineCss = 6,
    PressStart = 7,
    UnknownMenu = 0xFF,
}

impl Default for Menu {
    fn default() -> Self {
        Menu::InGame
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SubMenu {
    MainMenuSubmenu = 0,
    OnepModeSubmenu = 1,
    VsModeSubmenu = 2,
    TrophiesSubmenu = 3,
    OptionSubmenu = 4,
    DataSubmenu = 5,
    RegularMatchSubmenu = 6,
    EventMatchSubmenu = 7,
    OnlinePlaySubmenu = 8,
    StadiumSubmenu = 9,
    SpecialMeleeSubmenu = 12,
    CustomRulesSubmenu = 13,
    NameEntrySubmenu = 18,
    RumbleSubmenu = 19,
    SoundSubmenu = 20,
    ScreenDisplaySubmenu = 21,
    LanguageSelectSubmenu = 23,
    EraseDataSubmenu = 24,
    MultimanMeleeSubmenu = 33,
    OnlineCss = 0xFE,
    UnknownSubmenu = 0xFF,
}

impl Default for SubMenu {
    fn default() -> Self {
        SubMenu::UnknownSubmenu
    }
}

// =====================
// ControllerStatus / ControllerType / AttackState
// =====================

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ControllerStatus {
    ControllerHuman = 0,
    ControllerCpu = 1,
    ControllerUnplugged = 3,
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerType {
    /// Dolphin "standard" input (named pipe etc.)
    Standard,
    /// Wii U / Mayflash style GCN adapter
    GcnAdapter,
    /// Unplugged
    Unplugged,
}

#[pymethods]
impl ControllerType {
    /// Dolphin device code as a string: "6" (standard), "12" (GCN adapter), "0" (unplugged).
    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        match self {
            ControllerType::Standard => PyString::new_bound(py, "6"),
            ControllerType::GcnAdapter => PyString::new_bound(py, "12"),
            ControllerType::Unplugged => PyString::new_bound(py, "0"),
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AttackState {
    Windup = 0,
    Attacking = 1,
    Cooldown = 2,
    NotAttacking = 3,
}

// =====================
// Character
// =====================

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Character {
    Mario = 0x00,
    Fox = 0x01,
    CptFalcon = 0x02,
    Dk = 0x03,
    Kirby = 0x04,
    Bowser = 0x05,
    Link = 0x06,
    Sheik = 0x07,
    Ness = 0x08,
    Peach = 0x09,
    Popo = 0x0A,
    Nana = 0x0B,
    Pikachu = 0x0C,
    Samus = 0x0D,
    Yoshi = 0x0E,
    Jigglypuff = 0x0F,
    Mewtwo = 0x10,
    Luigi = 0x11,
    Marth = 0x12,
    Zelda = 0x13,
    Ylink = 0x14,
    Doc = 0x15,
    Falco = 0x16,
    Pichu = 0x17,
    Gameandwatch = 0x18,
    Ganondorf = 0x19,
    Roy = 0x1A,
    WireframeMale = 0x1D,
    WireframeFemale = 0x1E,
    GigaBowser = 0x1F,
    Sandbag = 0x20,
    UnknownCharacter = 0xFF,
}

#[pymethods]
impl Character {
    #[getter]
    fn value(&self) -> u8 {
        *self as u8
    }
}

impl Character {
    /// Direct "internal" character ID → enum.
    pub fn from_u8(val: u8) -> Self {
        match val {
            0x00 => Character::Mario,
            0x01 => Character::Fox,
            0x02 => Character::CptFalcon,
            0x03 => Character::Dk,
            0x04 => Character::Kirby,
            0x05 => Character::Bowser,
            0x06 => Character::Link,
            0x07 => Character::Sheik,
            0x08 => Character::Ness,
            0x09 => Character::Peach,
            0x0A => Character::Popo,
            0x0B => Character::Nana,
            0x0C => Character::Pikachu,
            0x0D => Character::Samus,
            0x0E => Character::Yoshi,
            0x0F => Character::Jigglypuff,
            0x10 => Character::Mewtwo,
            0x11 => Character::Luigi,
            0x12 => Character::Marth,
            0x13 => Character::Zelda,
            0x14 => Character::Ylink,
            0x15 => Character::Doc,
            0x16 => Character::Falco,
            0x17 => Character::Pichu,
            0x18 => Character::Gameandwatch,
            0x19 => Character::Ganondorf,
            0x1A => Character::Roy,
            0x1D => Character::WireframeMale,
            0x1E => Character::WireframeFemale,
            0x1F => Character::GigaBowser,
            0x20 => Character::Sandbag,
            _ => Character::UnknownCharacter,
        }
    }

    /// CSS (character select screen) ID → internal enum mapping (Python `to_internal`).
    pub fn from_css_id(val: u8) -> Self {
        match val {
            0x00 => Character::Doc,
            0x01 => Character::Mario,
            0x02 => Character::Luigi,
            0x03 => Character::Bowser,
            0x04 => Character::Peach,
            0x05 => Character::Yoshi,
            0x06 => Character::Dk,
            0x07 => Character::CptFalcon,
            0x08 => Character::Ganondorf,
            0x09 => Character::Falco,
            0x0A => Character::Fox,
            0x0B => Character::Ness,
            0x0C => Character::Popo,
            0x0D => Character::Kirby,
            0x0E => Character::Samus,
            0x0F => Character::Zelda,
            0x10 => Character::Link,
            0x11 => Character::Ylink,
            0x12 => Character::Pichu,
            0x13 => Character::Pikachu,
            0x14 => Character::Jigglypuff,
            0x15 => Character::Mewtwo,
            0x16 => Character::Gameandwatch,
            0x17 => Character::Marth,
            0x18 => Character::Roy,
            _ => Character::UnknownCharacter,
        }
    }

    /// Internal enum → CSS ID (Python `from_internal`).
    pub fn to_css_id(self) -> u8 {
        match self {
            Character::Doc => 0x00,
            Character::Mario => 0x01,
            Character::Luigi => 0x02,
            Character::Bowser => 0x03,
            Character::Peach => 0x04,
            Character::Yoshi => 0x05,
            Character::Dk => 0x06,
            Character::CptFalcon => 0x07,
            Character::Ganondorf => 0x08,
            Character::Falco => 0x09,
            Character::Fox => 0x0A,
            Character::Ness => 0x0B,
            Character::Popo => 0x0C,
            Character::Kirby => 0x0D,
            Character::Samus => 0x0E,
            Character::Zelda => 0x0F,
            Character::Link => 0x10,
            Character::Ylink => 0x11,
            Character::Pichu => 0x12,
            Character::Pikachu => 0x13,
            Character::Jigglypuff => 0x14,
            Character::Mewtwo => 0x15,
            Character::Gameandwatch => 0x16,
            Character::Marth => 0x17,
            Character::Roy => 0x18,
            _ => 0xFF,
        }
    }
}

impl Default for Character {
    fn default() -> Self {
        Character::UnknownCharacter
    }
}

// =====================
// Button
// =====================

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Button {
    ButtonA,
    ButtonB,
    ButtonX,
    ButtonY,
    ButtonZ,
    ButtonL,
    ButtonR,
    ButtonStart,
    ButtonDUp,
    ButtonDDown,
    ButtonDLeft,
    ButtonDRight,
    ButtonMain,
    ButtonC,
}

#[pymethods]
impl Button {
    /// Dolphin input string (matches Python Enum string values).
    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let s = match self {
            Button::ButtonA => "A",
            Button::ButtonB => "B",
            Button::ButtonX => "X",
            Button::ButtonY => "Y",
            Button::ButtonZ => "Z",
            Button::ButtonL => "L",
            Button::ButtonR => "R",
            Button::ButtonStart => "START",
            Button::ButtonDUp => "D_UP",
            Button::ButtonDDown => "D_DOWN",
            Button::ButtonDLeft => "D_LEFT",
            Button::ButtonDRight => "D_RIGHT",
            Button::ButtonMain => "MAIN",
            Button::ButtonC => "C",
        };
        PyString::new_bound(py, s)
    }
}

// =====================
// Action
// =====================

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum Action {
    // Death states
    DeadDown = 0x000,
    DeadLeft = 0x001,
    DeadRight = 0x002,
    DeadUp = 0x003,
    DeadFlyStar = 0x004,
    DeadFlyStarIce = 0x005,
    DeadFly = 0x006,
    DeadFlySplatter = 0x007,
    DeadFlySplatterFlat = 0x008,
    DeadFlySplatterIce = 0x009,
    DeadFlySplatterFlatIce = 0x00A,

    // Misc / spawn
    NothingState = 0x00B,
    OnHaloDescent = 0x00C,
    OnHaloWait = 0x00D,

    // Ground / locomotion
    Standing = 0x00E,
    WalkSlow = 0x00F,
    WalkMiddle = 0x010,
    WalkFast = 0x011,
    Turning = 0x012,
    TurningRun = 0x013,
    Dashing = 0x014,
    Running = 0x015,
    RunDirect = 0x016,
    RunBrake = 0x017,

    // Jumps / air
    KneeBend = 0x018,
    JumpingForward = 0x019,
    JumpingBackward = 0x01A,
    JumpingArialForward = 0x01B,
    JumpingArialBackward = 0x01C,
    Falling = 0x01D,
    FallingForward = 0x01E,
    FallingBackward = 0x01F,
    FallingAerial = 0x020,
    FallingAerialForward = 0x021,
    FallingAerialBackward = 0x022,
    DeadFall = 0x023,
    SpecialFallForward = 0x024,
    SpecialFallBack = 0x025,
    Tumbling = 0x026,

    // Crouch / landing
    CrouchStart = 0x027,
    Crouching = 0x028,
    CrouchEnd = 0x029,
    Landing = 0x02A,
    LandingSpecial = 0x02B,

    // Jab / tilts / smashes / aerials
    NeutralAttack1 = 0x02C,
    NeutralAttack2 = 0x02D,
    NeutralAttack3 = 0x02E,
    LoopingAttackStart = 0x02F,
    LoopingAttackMiddle = 0x030,
    LoopingAttackEnd = 0x031,
    DashAttack = 0x032,
    FtiltHigh = 0x033,
    FtiltHighMid = 0x034,
    FtiltMid = 0x035,
    FtiltLowMid = 0x036,
    FtiltLow = 0x037,
    Uptilt = 0x038,
    Downtilt = 0x039,
    FsmashHigh = 0x03A,
    FsmashMidHigh = 0x03B,
    FsmashMid = 0x03C,
    FsmashMidLow = 0x03D,
    FsmashLow = 0x03E,
    Upsmash = 0x03F,
    Downsmash = 0x040,
    Nair = 0x041,
    Fair = 0x042,
    Bair = 0x043,
    Uair = 0x044,
    Dair = 0x045,
    NairLanding = 0x046,
    FairLanding = 0x047,
    BairLanding = 0x048,
    UairLanding = 0x049,
    DairLanding = 0x04A,

    // Damage (ground/air/tumble)
    DamageHigh1 = 0x04B,
    DamageHigh2 = 0x04C,
    DamageHigh3 = 0x04D,
    DamageNeutral1 = 0x04E,
    DamageNeutral2 = 0x04F,
    DamageNeutral3 = 0x050,
    DamageLow1 = 0x051,
    DamageLow2 = 0x052,
    DamageLow3 = 0x053,
    DamageAir1 = 0x054,
    DamageAir2 = 0x055,
    DamageAir3 = 0x056,
    DamageFlyHigh = 0x057,
    DamageFlyNeutral = 0x058,
    DamageFlyLow = 0x059,
    DamageFlyTop = 0x05A,
    DamageFlyRoll = 0x05B,

    // Item interactions
    ItemPickupLight = 0x05C,
    ItemPickupHeavy = 0x05D,
    ItemThrowLightForward = 0x05E,
    ItemThrowLightBack = 0x05F,
    ItemThrowLightHigh = 0x060,
    ItemThrowLightLow = 0x061,
    ItemThrowLightDash = 0x062,
    ItemThrowLightDrop = 0x063,
    ItemThrowLightAirForward = 0x064,
    ItemThrowLightAirBack = 0x065,
    ItemThrowLightAirHigh = 0x066,
    ItemThrowLightAirLow = 0x067,
    ItemThrowHeavyForward = 0x068,
    ItemThrowHeavyBack = 0x069,
    ItemThrowHeavyHigh = 0x06A,
    ItemThrowHeavyLow = 0x06B,
    ItemThrowLightSmashForward = 0x06C,
    ItemThrowLightSmashBack = 0x06D,
    ItemThrowLightSmashUp = 0x06E,
    ItemThrowLightSmashDown = 0x06F,
    ItemThrowLightAirSmashForward = 0x070,
    ItemThrowLightAirSmashBack = 0x071,
    ItemThrowLightAirSmashHigh = 0x072,
    ItemThrowLightAirSmashLow = 0x073,
    ItemThrowHeavyAirSmashForward = 0x074,
    ItemThrowHeavyAirSmashBack = 0x075,
    ItemThrowHeavyAirSmashHigh = 0x076,
    ItemThrowHeavyAirSmashLow = 0x077,
    BeamSwordSwing1 = 0x078,
    BeamSwordSwing2 = 0x079,
    BeamSwordSwing3 = 0x07A,
    BeamSwordSwing4 = 0x07B,
    BatSwing1 = 0x07C,
    BatSwing2 = 0x07D,
    BatSwing3 = 0x07E,
    BatSwing4 = 0x07F,
    ParasolSwing1 = 0x080,
    ParasolSwing2 = 0x081,
    ParasolSwing3 = 0x082,
    ParasolSwing4 = 0x083,
    FanSwing1 = 0x084,
    FanSwing2 = 0x085,
    FanSwing3 = 0x086,
    FanSwing4 = 0x087,
    StarRodSwing1 = 0x088,
    StarRodSwing2 = 0x089,
    StarRodSwing3 = 0x08A,
    StarRodSwing4 = 0x08B,
    LipStickSwing1 = 0x08C,
    LipStickSwing2 = 0x08D,
    LipStickSwing3 = 0x08E,
    LipStickSwing4 = 0x08F,
    ItemParasolOpen = 0x090,
    ItemParasolFall = 0x091,
    ItemParasolFallSpecial = 0x092,
    ItemParasolDamageFall = 0x093,

    // Guns / flowers / screws / scope
    GunShoot = 0x094,
    GunShootAir = 0x095,
    GunShootEmpty = 0x096,
    GunShootAirEmpty = 0x097,
    FireFlowerShoot = 0x098,
    FireFlowerShootAir = 0x099,
    ItemScrew = 0x09A,
    ItemScrewAir = 0x09B,
    DamageScrew = 0x09C,
    DamageScrewAir = 0x09D,
    ItemScopeStart = 0x09E,
    ItemScopeRapid = 0x09F,
    ItemScopeFire = 0x0A0,
    ItemScopeEnd = 0x0A1,
    ItemScopeAirStart = 0x0A2,
    ItemScopeAirRapid = 0x0A3,
    ItemScopeAirFire = 0x0A4,
    ItemScopeAirEnd = 0x0A5,
    ItemScopeStartEmpty = 0x0A6,
    ItemScopeRapidEmpty = 0x0A7,
    ItemScopeFireEmpty = 0x0A8,
    ItemScopeEndEmpty = 0x0A9,
    ItemScopeAirStartEmpty = 0x0AA,
    ItemScopeAirRapidEmpty = 0x0AB,
    ItemScopeAirFireEmpty = 0x0AC,
    ItemScopeAirEndEmpty = 0x0AD,

    // Lift (platform) etc.
    LiftWait = 0x0AE,
    LiftWalk1 = 0x0AF,
    LiftWalk2 = 0x0B0,
    LiftTurn = 0x0B1,

    // Shield
    ShieldStart = 0x0B2,
    Shield = 0x0B3,
    ShieldRelease = 0x0B4,
    ShieldStun = 0x0B5,
    ShieldReflect = 0x0B6,

    // Tech / getup (UP “facing up” branch)
    TechMissUp = 0x0B7,
    LyingGroundUp = 0x0B8,
    LyingGroundUpHit = 0x0B9,
    GroundGetup = 0x0BA,
    GroundAttackUp = 0x0BB,
    GroundRollForwardUp = 0x0BC,
    GroundRollBackwardUp = 0x0BD,
    GroundSpotUp = 0x0BE,

    // Tech / getup (DOWN branch)
    TechMissDown = 0x0BF,
    LyingGroundDown = 0x0C0,
    DamageGround = 0x0C1,
    NeutralGetup = 0x0C2,
    GetupAttack = 0x0C3,
    GroundRollForwardDown = 0x0C4,
    GroundRollBackwardDown = 0x0C5,
    GroundRollSpotDown = 0x0C6,

    // Tech actual
    NeutralTech = 0x0C7,
    ForwardTech = 0x0C8,
    BackwardTech = 0x0C9,
    WallTech = 0x0CA,
    WallTechJump = 0x0CB,
    CeilingTech = 0x0CC,

    // Shield break sequence
    ShieldBreakFly = 0x0CD,
    ShieldBreakFall = 0x0CE,
    ShieldBreakDownU = 0x0CF,
    ShieldBreakDownD = 0x0D0,
    ShieldBreakStandU = 0x0D1,
    ShieldBreakStandD = 0x0D2,
    ShieldBreakTeeter = 0x0D3,

    // Grabs / throws
    Grab = 0x0D4,
    GrabPulling = 0x0D5,
    GrabRunning = 0x0D6,
    GrabRunningPulling = 0x0D7,
    GrabWait = 0x0D8,
    GrabPummel = 0x0D9,
    GrabBreak = 0x0DA,
    ThrowForward = 0x0DB,
    ThrowBack = 0x0DC,
    ThrowUp = 0x0DD,
    ThrowDown = 0x0DE,
    GrabPullingHigh = 0x0DF,
    GrabbedWaitHigh = 0x0E0,
    PummeledHigh = 0x0E1,
    GrabPull = 0x0E2,
    Grabbed = 0x0E3,
    GrabPummeled = 0x0E4,
    GrabEscape = 0x0E5,
    GrabJump = 0x0E6,
    GrabNeck = 0x0E7,
    GrabFoot = 0x0E8,

    // Defensive (roll/dodge/airdodge)
    RollForward = 0x0E9,
    RollBackward = 0x0EA,
    Spotdodge = 0x0EB,
    Airdodge = 0x0EC,

    // Rebound/bounce (uncertain)
    ReboundStop = 0x0ED,
    Rebound = 0x0EE,

    // Thrown (you being thrown)
    ThrownForward = 0x0EF,
    ThrownBack = 0x0F0,
    ThrownUp = 0x0F1,
    ThrownDown = 0x0F2,
    ThrownDown2 = 0x0F3,

    // Edge / platform
    PlatformDrop = 0x0F4,
    EdgeTeeteringStart = 0x0F5,
    EdgeTeetering = 0x0F6,
    BounceWall = 0x0F7,
    BounceCeiling = 0x0F8,
    BumpWall = 0x0F9,
    BumpCieling = 0x0FA,
    SlidingOffEdge = 0x0FB,
    EdgeCatching = 0x0FC,
    EdgeHanging = 0x0FD,
    EdgeGetupSlow = 0x0FE,
    EdgeGetupQuick = 0x0FF,
    EdgeAttackSlow = 0x100,
    EdgeAttackQuick = 0x101,
    EdgeRollSlow = 0x102,
    EdgeRollQuick = 0x103,
    EdgeJump1Slow = 0x104,
    EdgeJump2Slow = 0x105,
    EdgeJump1Quick = 0x106,
    EdgeJump2Quick = 0x107,

    // Taunts
    TauntRight = 0x108,
    TauntLeft = 0x109,

    // Shouldered (DK carry etc.)
    ShoulderedWait = 0x10A,
    ShoulderedWalkSlow = 0x10B,
    ShoulderedWalkMiddle = 0x10C,
    ShoulderedWalkFast = 0x10D,
    ShoulderedTurn = 0x10E,
    ThrownFf = 0x10F,
    ThrownFb = 0x110,
    ThrownFHigh = 0x111,
    ThrownFLow = 0x112,

    // Captures (Yoshi/DK/Kirby/etc.)
    CaptureCaptain = 0x113,
    CaptureYoshi = 0x114,
    YoshiEgg = 0x115,
    CaptureKoopa = 0x116,
    CaptureDamageKoopa = 0x117,
    CaptureWaitKoopa = 0x118,
    ThrownKoopaF = 0x119,
    ThrownKoopaB = 0x11A,
    CaptureKoopaAir = 0x11B,
    CaptureDamageKoopaAir = 0x11C,
    CaptureWaitKoopaAir = 0x11D,
    ThrownKoopaAirF = 0x11E,
    ThrownKoopaAirB = 0x11F,
    CaptureKirby = 0x120,
    CaptureWaitKirby = 0x121,
    ThrownKirbyStar = 0x122,
    ThrownCopyStar = 0x123,
    ThrownKirby = 0x124,

    // Barrels / bury / sleep / disable
    BarrelWait = 0x125,
    Bury = 0x126,
    BuryWait = 0x127,
    BuryJump = 0x128,
    DamageSong = 0x129,
    DamageSongWait = 0x12A,
    DamageSongRv = 0x12B,
    DamageBind = 0x12C,
    CaptureMewtwo = 0x12D,
    CaptureMewtwoAir = 0x12E,
    ThrownMewtwo = 0x12F,
    ThrownMewtwoAir = 0x130,

    // Warp star
    WarpStarJump = 0x131,
    WarpStapFall = 0x132,

    // Hammer
    HammerWait = 0x133,
    HammerWalk = 0x134,
    HammerTurn = 0x135,
    HammerKneeBend = 0x136,
    HammerFall = 0x137,
    HammerJump = 0x138,
    HammerLanding = 0x139,

    // Mushrooms
    KinokoGiantStart = 0x13A,
    KinokoGiantStartAir = 0x13B,
    KinokoGiantEnd = 0x13C,
    KinokoGiantEndAir = 0x13D,
    KinokoSmallStart = 0x13E,
    KinokoSmallStartAir = 0x13F,
    KinokoSmallEnd = 0x140,
    KinokoSmallEndAir = 0x141,

    // Match start
    Entry = 0x142,
    EntryStart = 0x143,
    EntryEnd = 0x144,

    // Ice
    DamageIce = 0x145,
    DamageIceJump = 0x146,

    // Master/Crazy Hand captures
    CaptureMasterhand = 0x147,
    CaptureDamageMasterhand = 0x148,
    CaptureWaitMasterhand = 0x149,
    ThrownMasterhand = 0x14A,
    CaptureKirbyYoshi = 0x14B,
    KirbyYoshiEgg = 0x14C,
    CaptureLeaDead = 0x14D,
    CaptureLikeLike = 0x14E,

    // Misc
    DownReflect = 0x14F,
    CaptureCrazyhand = 0x150,
    CaptureDamageCrazyhand = 0x151,
    CaptureWaitCrazyhand = 0x152,
    ThrownCrazyHand = 0x153,
    BarrelCannonWait = 0x154,
    LaserGunPull = 0x155,

    // Neutral-B charge/attack (ground/air)
    NeutralBCharging = 0x156,
    NeutralBAttacking = 0x157,
    NeutralBFullCharge = 0x158,
    WaitItem = 0x159,
    NeutralBChargingAir = 0x15A,
    NeutralBAttackingAir = 0x15B,
    NeutralBFullChargeAir = 0x15C,

    // Marth sword dance (ground/air)
    SwordDance1 = 0x15D,
    SwordDance4Low = 0x165,
    SwordDance1Air = 0x166,
    SwordDance2HighAir = 0x167,
    SwordDance3MidAir = 0x16A,
    SwordDance3LowAir = 0x16B,

    // Fox shine / Up-B
    DownBGroundStart = 0x168,
    DownBGround = 0x169,
    ShineTurn = 0x16C,
    DownBStun = 0x16D,
    DownBAir = 0x16E,
    UpBGround = 0x16F,
    ShineReleaseAir = 0x170,

    // Fox Illusion / Firefox
    FoxIllusionStart = 0x15E,
    FoxIllusion = 0x15F,
    FoxIllusionShortened = 0x160,
    FirefoxWaitGround = 0x161,
    FirefoxWaitAir = 0x162,
    FirefoxGround = 0x163,
    FirefoxAir = 0x164,

    // Marth / Ness misc
    MarthCounter = 0x171,
    ParasolFalling = 0x172,
    MarthCounterFalling = 0x173,
    NessSheildStart = 0x174,
    NessSheildAir = 0x175,
    Zitabata = 0x176,
    NessSheildAirEnd = 0x177,

    // Koopa throw end variants
    ThrownKoopaEndF = 0x178,
    ThrownKoopaEndB = 0x179,
    CaptureKoopaAirHit = 0x17A,
    ThrownKoopaAirEndF = 0x17B,
    ThrownKoopaAirEndB = 0x17C,

    // Kirby drink/spit shot
    ThrownKirbyDrinkSShot = 0x17D,
    ThrownKirbySpitSShot = 0x17E,

    // DK Ground Pound
    DkGroundPoundStart = 0x17F,
    DkGroundPound = 0x180,
    DkGroundPoundEnd = 0x181,

    // Kirby blade / stone
    KirbyBladeGround = 0x184,
    KirbyBladeUp = 0x185,
    KirbyBladeApex = 0x186,
    KirbyBladeDown = 0x187,
    KirbyStoneFormingGround = 0x189,
    KirbyStoneResting = 0x18A,
    KirbyStoneRelease = 0x18B,
    KirbyStoneFormingAir = 0x18C,
    KirbyStoneFalling = 0x18D,

    UnknownAnimation = 0xFFFF,
}

#[pymethods]
impl Action {
    #[getter]
    fn value(&self) -> u16 {
        *self as u16
    }
}

impl Action {
    pub fn from_u16(val: u16) -> Self {
        use Action::*;
        match val {
            0x000 => DeadDown,
            0x001 => DeadLeft,
            0x002 => DeadRight,
            0x003 => DeadUp,
            0x004 => DeadFlyStar,
            0x005 => DeadFlyStarIce,
            0x006 => DeadFly,
            0x007 => DeadFlySplatter,
            0x008 => DeadFlySplatterFlat,
            0x009 => DeadFlySplatterIce,
            0x00A => DeadFlySplatterFlatIce,
            0x00B => NothingState,
            0x00C => OnHaloDescent,
            0x00D => OnHaloWait,
            0x00E => Standing,
            0x00F => WalkSlow,
            0x010 => WalkMiddle,
            0x011 => WalkFast,
            0x012 => Turning,
            0x013 => TurningRun,
            0x014 => Dashing,
            0x015 => Running,
            0x016 => RunDirect,
            0x017 => RunBrake,
            0x018 => KneeBend,
            0x019 => JumpingForward,
            0x01A => JumpingBackward,
            0x01B => JumpingArialForward,
            0x01C => JumpingArialBackward,
            0x01D => Falling,
            0x01E => FallingForward,
            0x01F => FallingBackward,
            0x020 => FallingAerial,
            0x021 => FallingAerialForward,
            0x022 => FallingAerialBackward,
            0x023 => DeadFall,
            0x024 => SpecialFallForward,
            0x025 => SpecialFallBack,
            0x026 => Tumbling,
            0x027 => CrouchStart,
            0x028 => Crouching,
            0x029 => CrouchEnd,
            0x02A => Landing,
            0x02B => LandingSpecial,
            0x02C => NeutralAttack1,
            0x02D => NeutralAttack2,
            0x02E => NeutralAttack3,
            0x02F => LoopingAttackStart,
            0x030 => LoopingAttackMiddle,
            0x031 => LoopingAttackEnd,
            0x032 => DashAttack,
            0x033 => FtiltHigh,
            0x034 => FtiltHighMid,
            0x035 => FtiltMid,
            0x036 => FtiltLowMid,
            0x037 => FtiltLow,
            0x038 => Uptilt,
            0x039 => Downtilt,
            0x03A => FsmashHigh,
            0x03B => FsmashMidHigh,
            0x03C => FsmashMid,
            0x03D => FsmashMidLow,
            0x03E => FsmashLow,
            0x03F => Upsmash,
            0x040 => Downsmash,
            0x041 => Nair,
            0x042 => Fair,
            0x043 => Bair,
            0x044 => Uair,
            0x045 => Dair,
            0x046 => NairLanding,
            0x047 => FairLanding,
            0x048 => BairLanding,
            0x049 => UairLanding,
            0x04A => DairLanding,
            0x04B => DamageHigh1,
            0x04C => DamageHigh2,
            0x04D => DamageHigh3,
            0x04E => DamageNeutral1,
            0x04F => DamageNeutral2,
            0x050 => DamageNeutral3,
            0x051 => DamageLow1,
            0x052 => DamageLow2,
            0x053 => DamageLow3,
            0x054 => DamageAir1,
            0x055 => DamageAir2,
            0x056 => DamageAir3,
            0x057 => DamageFlyHigh,
            0x058 => DamageFlyNeutral,
            0x059 => DamageFlyLow,
            0x05A => DamageFlyTop,
            0x05B => DamageFlyRoll,
            0x05C => ItemPickupLight,
            0x05D => ItemPickupHeavy,
            0x05E => ItemThrowLightForward,
            0x05F => ItemThrowLightBack,
            0x060 => ItemThrowLightHigh,
            0x061 => ItemThrowLightLow,
            0x062 => ItemThrowLightDash,
            0x063 => ItemThrowLightDrop,
            0x064 => ItemThrowLightAirForward,
            0x065 => ItemThrowLightAirBack,
            0x066 => ItemThrowLightAirHigh,
            0x067 => ItemThrowLightAirLow,
            0x068 => ItemThrowHeavyForward,
            0x069 => ItemThrowHeavyBack,
            0x06A => ItemThrowHeavyHigh,
            0x06B => ItemThrowHeavyLow,
            0x06C => ItemThrowLightSmashForward,
            0x06D => ItemThrowLightSmashBack,
            0x06E => ItemThrowLightSmashUp,
            0x06F => ItemThrowLightSmashDown,
            0x070 => ItemThrowLightAirSmashForward,
            0x071 => ItemThrowLightAirSmashBack,
            0x072 => ItemThrowLightAirSmashHigh,
            0x073 => ItemThrowLightAirSmashLow,
            0x074 => ItemThrowHeavyAirSmashForward,
            0x075 => ItemThrowHeavyAirSmashBack,
            0x076 => ItemThrowHeavyAirSmashHigh,
            0x077 => ItemThrowHeavyAirSmashLow,
            0x078 => BeamSwordSwing1,
            0x079 => BeamSwordSwing2,
            0x07A => BeamSwordSwing3,
            0x07B => BeamSwordSwing4,
            0x07C => BatSwing1,
            0x07D => BatSwing2,
            0x07E => BatSwing3,
            0x07F => BatSwing4,
            0x080 => ParasolSwing1,
            0x081 => ParasolSwing2,
            0x082 => ParasolSwing3,
            0x083 => ParasolSwing4,
            0x084 => FanSwing1,
            0x085 => FanSwing2,
            0x086 => FanSwing3,
            0x087 => FanSwing4,
            0x088 => StarRodSwing1,
            0x089 => StarRodSwing2,
            0x08A => StarRodSwing3,
            0x08B => StarRodSwing4,
            0x08C => LipStickSwing1,
            0x08D => LipStickSwing2,
            0x08E => LipStickSwing3,
            0x08F => LipStickSwing4,
            0x090 => ItemParasolOpen,
            0x091 => ItemParasolFall,
            0x092 => ItemParasolFallSpecial,
            0x093 => ItemParasolDamageFall,
            0x094 => GunShoot,
            0x095 => GunShootAir,
            0x096 => GunShootEmpty,
            0x097 => GunShootAirEmpty,
            0x098 => FireFlowerShoot,
            0x099 => FireFlowerShootAir,
            0x09A => ItemScrew,
            0x09B => ItemScrewAir,
            0x09C => DamageScrew,
            0x09D => DamageScrewAir,
            0x09E => ItemScopeStart,
            0x09F => ItemScopeRapid,
            0x0A0 => ItemScopeFire,
            0x0A1 => ItemScopeEnd,
            0x0A2 => ItemScopeAirStart,
            0x0A3 => ItemScopeAirRapid,
            0x0A4 => ItemScopeAirFire,
            0x0A5 => ItemScopeAirEnd,
            0x0A6 => ItemScopeStartEmpty,
            0x0A7 => ItemScopeRapidEmpty,
            0x0A8 => ItemScopeFireEmpty,
            0x0A9 => ItemScopeEndEmpty,
            0x0AA => ItemScopeAirStartEmpty,
            0x0AB => ItemScopeAirRapidEmpty,
            0x0AC => ItemScopeAirFireEmpty,
            0x0AD => ItemScopeAirEndEmpty,
            0x0AE => LiftWait,
            0x0AF => LiftWalk1,
            0x0B0 => LiftWalk2,
            0x0B1 => LiftTurn,
            0x0B2 => ShieldStart,
            0x0B3 => Shield,
            0x0B4 => ShieldRelease,
            0x0B5 => ShieldStun,
            0x0B6 => ShieldReflect,
            0x0B7 => TechMissUp,
            0x0B8 => LyingGroundUp,
            0x0B9 => LyingGroundUpHit,
            0x0BA => GroundGetup,
            0x0BB => GroundAttackUp,
            0x0BC => GroundRollForwardUp,
            0x0BD => GroundRollBackwardUp,
            0x0BE => GroundSpotUp,
            0x0BF => TechMissDown,
            0x0C0 => LyingGroundDown,
            0x0C1 => DamageGround,
            0x0C2 => NeutralGetup,
            0x0C3 => GetupAttack,
            0x0C4 => GroundRollForwardDown,
            0x0C5 => GroundRollBackwardDown,
            0x0C6 => GroundRollSpotDown,
            0x0C7 => NeutralTech,
            0x0C8 => ForwardTech,
            0x0C9 => BackwardTech,
            0x0CA => WallTech,
            0x0CB => WallTechJump,
            0x0CC => CeilingTech,
            0x0CD => ShieldBreakFly,
            0x0CE => ShieldBreakFall,
            0x0CF => ShieldBreakDownU,
            0x0D0 => ShieldBreakDownD,
            0x0D1 => ShieldBreakStandU,
            0x0D2 => ShieldBreakStandD,
            0x0D3 => ShieldBreakTeeter,
            0x0D4 => Grab,
            0x0D5 => GrabPulling,
            0x0D6 => GrabRunning,
            0x0D7 => GrabRunningPulling,
            0x0D8 => GrabWait,
            0x0D9 => GrabPummel,
            0x0DA => GrabBreak,
            0x0DB => ThrowForward,
            0x0DC => ThrowBack,
            0x0DD => ThrowUp,
            0x0DE => ThrowDown,
            0x0DF => GrabPullingHigh,
            0x0E0 => GrabbedWaitHigh,
            0x0E1 => PummeledHigh,
            0x0E2 => GrabPull,
            0x0E3 => Grabbed,
            0x0E4 => GrabPummeled,
            0x0E5 => GrabEscape,
            0x0E6 => GrabJump,
            0x0E7 => GrabNeck,
            0x0E8 => GrabFoot,
            0x0E9 => RollForward,
            0x0EA => RollBackward,
            0x0EB => Spotdodge,
            0x0EC => Airdodge,
            0x0ED => ReboundStop,
            0x0EE => Rebound,
            0x0EF => ThrownForward,
            0x0F0 => ThrownBack,
            0x0F1 => ThrownUp,
            0x0F2 => ThrownDown,
            0x0F3 => ThrownDown2,
            0x0F4 => PlatformDrop,
            0x0F5 => EdgeTeeteringStart,
            0x0F6 => EdgeTeetering,
            0x0F7 => BounceWall,
            0x0F8 => BounceCeiling,
            0x0F9 => BumpWall,
            0x0FA => BumpCieling,
            0x0FB => SlidingOffEdge,
            0x0FC => EdgeCatching,
            0x0FD => EdgeHanging,
            0x0FE => EdgeGetupSlow,
            0x0FF => EdgeGetupQuick,
            0x100 => EdgeAttackSlow,
            0x101 => EdgeAttackQuick,
            0x102 => EdgeRollSlow,
            0x103 => EdgeRollQuick,
            0x104 => EdgeJump1Slow,
            0x105 => EdgeJump2Slow,
            0x106 => EdgeJump1Quick,
            0x107 => EdgeJump2Quick,
            0x108 => TauntRight,
            0x109 => TauntLeft,
            0x10A => ShoulderedWait,
            0x10B => ShoulderedWalkSlow,
            0x10C => ShoulderedWalkMiddle,
            0x10D => ShoulderedWalkFast,
            0x10E => ShoulderedTurn,
            0x10F => ThrownFf,
            0x110 => ThrownFb,
            0x111 => ThrownFHigh,
            0x112 => ThrownFLow,
            0x113 => CaptureCaptain,
            0x114 => CaptureYoshi,
            0x115 => YoshiEgg,
            0x116 => CaptureKoopa,
            0x117 => CaptureDamageKoopa,
            0x118 => CaptureWaitKoopa,
            0x119 => ThrownKoopaF,
            0x11A => ThrownKoopaB,
            0x11B => CaptureKoopaAir,
            0x11C => CaptureDamageKoopaAir,
            0x11D => CaptureWaitKoopaAir,
            0x11E => ThrownKoopaAirF,
            0x11F => ThrownKoopaAirB,
            0x120 => CaptureKirby,
            0x121 => CaptureWaitKirby,
            0x122 => ThrownKirbyStar,
            0x123 => ThrownCopyStar,
            0x124 => ThrownKirby,
            0x125 => BarrelWait,
            0x126 => Bury,
            0x127 => BuryWait,
            0x128 => BuryJump,
            0x129 => DamageSong,
            0x12A => DamageSongWait,
            0x12B => DamageSongRv,
            0x12C => DamageBind,
            0x12D => CaptureMewtwo,
            0x12E => CaptureMewtwoAir,
            0x12F => ThrownMewtwo,
            0x130 => ThrownMewtwoAir,
            0x131 => WarpStarJump,
            0x132 => WarpStapFall,
            0x133 => HammerWait,
            0x134 => HammerWalk,
            0x135 => HammerTurn,
            0x136 => HammerKneeBend,
            0x137 => HammerFall,
            0x138 => HammerJump,
            0x139 => HammerLanding,
            0x13A => KinokoGiantStart,
            0x13B => KinokoGiantStartAir,
            0x13C => KinokoGiantEnd,
            0x13D => KinokoGiantEndAir,
            0x13E => KinokoSmallStart,
            0x13F => KinokoSmallStartAir,
            0x140 => KinokoSmallEnd,
            0x141 => KinokoSmallEndAir,
            0x142 => Entry,
            0x143 => EntryStart,
            0x144 => EntryEnd,
            0x145 => DamageIce,
            0x146 => DamageIceJump,
            0x147 => CaptureMasterhand,
            0x148 => CaptureDamageMasterhand,
            0x149 => CaptureWaitMasterhand,
            0x14A => ThrownMasterhand,
            0x14B => CaptureKirbyYoshi,
            0x14C => KirbyYoshiEgg,
            0x14D => CaptureLeaDead,
            0x14E => CaptureLikeLike,
            0x14F => DownReflect,
            0x150 => CaptureCrazyhand,
            0x151 => CaptureDamageCrazyhand,
            0x152 => CaptureWaitCrazyhand,
            0x153 => ThrownCrazyHand,
            0x154 => BarrelCannonWait,
            0x155 => LaserGunPull,
            0x156 => NeutralBCharging,
            0x157 => NeutralBAttacking,
            0x158 => NeutralBFullCharge,
            0x159 => WaitItem,
            0x15A => NeutralBChargingAir,
            0x15B => NeutralBAttackingAir,
            0x15C => NeutralBFullChargeAir,
            0x15D => SwordDance1,
            0x15E => FoxIllusionStart,
            0x15F => FoxIllusion,
            0x160 => FoxIllusionShortened,
            0x161 => FirefoxWaitGround,
            0x162 => FirefoxWaitAir,
            0x163 => FirefoxGround,
            0x164 => FirefoxAir,
            0x165 => SwordDance4Low,
            0x166 => SwordDance1Air,
            0x167 => SwordDance2HighAir,
            0x168 => DownBGroundStart,
            0x169 => DownBGround,
            0x16A => SwordDance3MidAir,
            0x16B => SwordDance3LowAir,
            0x16C => ShineTurn,
            0x16D => DownBStun,  // (DownBStun)
            0x16E => DownBAir,
            0x16F => UpBGround,
            0x170 => ShineReleaseAir,
            0x171 => MarthCounter,
            0x172 => ParasolFalling,
            0x173 => MarthCounterFalling,
            0x174 => NessSheildStart,
            0x175 => NessSheildAir,
            0x176 => Zitabata,
            0x177 => NessSheildAirEnd,
            0x178 => ThrownKoopaEndF,
            0x179 => ThrownKoopaEndB,
            0x17A => CaptureKoopaAirHit,
            0x17B => ThrownKoopaAirEndF,
            0x17C => ThrownKoopaAirEndB,
            0x17D => ThrownKirbyDrinkSShot,
            0x17E => ThrownKirbySpitSShot,
            0x17F => DkGroundPoundStart,
            0x180 => DkGroundPound,
            0x181 => DkGroundPoundEnd,
            0x184 => KirbyBladeGround,
            0x185 => KirbyBladeUp,
            0x186 => KirbyBladeApex,
            0x187 => KirbyBladeDown,
            0x189 => KirbyStoneFormingGround,
            0x18A => KirbyStoneResting,
            0x18B => KirbyStoneRelease,
            0x18C => KirbyStoneFormingAir,
            0x18D => KirbyStoneFalling,
            _ => UnknownAnimation,
        }
    }
}

impl Default for Action {
    fn default() -> Self {
        Action::UnknownAnimation
    }
}

// =====================
// ProjectileType
// =====================

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum ProjectileType {
    BobOmb = 0x06,
    MrSaturn = 0x07,
    Beamsword = 0x0C,
    MarioFireball = 0x30,
    DrMarioCapsule = 0x31,
    KirbyCutter = 0x32,
    KirbyHammer = 0x33,
    FoxLaser = 0x36,
    FalcoLaser = 0x37,
    FoxShadow = 0x38,
    FalcoShadow = 0x39,
    LinkBomb = 0x3A,
    YlinkBomb = 0x3B,
    LinkBoomerang = 0x3C,
    YlinkBoomerang = 0x3D,
    LinkHookshot = 0x3E,
    YlinkHookshot = 0x3F,
    Arrow = 0x40,
    FireArrow = 0x41,
    PkFire = 0x42,
    PkFlash1 = 0x43,
    PkFlash2 = 0x44,
    PkThunderHead = 0x45,
    PkThunderTail1 = 0x46,
    PkThunderTail2 = 0x47,
    PkThunderTail3 = 0x48,
    PkThunderTail4 = 0x49,
    LinkArrow = 0x4C,
    YlinkArrow = 0x4D,
    PkFlashExplosion = 0x4E,
    NeedleThrown = 0x4F,
    PikachuThunder = 0x51,
    PichuThunder = 0x52,
    MarioCape = 0x53,
    DrMarioCape = 0x54,
    SheikSmoke = 0x55,
    YoshiEggThrown = 0x56,
    YoshiTongue = 0x57,
    YoshiStar = 0x58,
    PikachuThunderjolt1 = 0x59,
    PikachuThunderjolt2 = 0x5A,
    PichuThunderjolt1 = 0x5B,
    PichuThunderjolt2 = 0x5C,
    SamusBomb = 0x5D,
    SamusChargeBeam = 0x5E,
    SamusMissle = 0x5F,
    SamusGrappleBeam = 0x60,
    SheikChain = 0x61,
    Turnip = 0x63,
    BowserFlame = 0x64,
    NessBatt = 0x65,
    NessYoyo = 0x66,
    PeachParasol = 0x67,
    LuigiFire = 0x69,
    IceBlock = 0x6A,
    IcBlizzard = 0x6B,
    ZeldaFire = 0x6C,
    ZeldaFireExplosion = 0x6D,
    MewtoDisable = 0x6E,
    ToadSpore = 0x6F,
    Shadowball = 0x70,
    IcUpB = 0x71,
    Pesticide = 0x72,
    Manhole = 0x73,
    GwFire = 0x74,
    Parachute = 0x75,
    Turtle = 0x76,
    Sperky = 0x77,
    Judge = 0x78,
    Sausage = 0x7A,
    YlinkMilk = 0x7B,
    Firefighter = 0x7C,

    // Kirby copy abilities
    KirbyMarioFire = 0x82,
    KirbyDrMarioFire = 0x83,
    KirbyLuigiFire = 0x84,
    KirbyIcBlock = 0x85,
    KirbyToadSpore = 0x87,
    KirbyFoxLaser = 0x88,
    KirbyFalcoLaser = 0x89,
    KirbyLinkArrow = 0x8C,
    KirbyYlinkArrow = 0x8D,
    KirbyLinkArrow2 = 0x8E,
    KirbyYlinkArrow2 = 0x8F,
    KirbyShadowball = 0x90,
    KirbyPkFlash = 0x91,
    KirbyPkFlashExplosion = 0x92,
    KirbyPikachuThunderjolt1 = 0x93,
    KirbyPikachuThunderjolt2 = 0x94,
    KirbyPichuThunderjolt1 = 0x95,
    KirbyPichuThunderjolt2 = 0x96,
    KirbySamusChargeshot = 0x97,
    KirbySheikNeedleThrown = 0x98,
    KirbySheikNeedleGround = 0x99,
    KirbyBowserFlame = 0x9A,
    KirbySausage = 0x9B,
    KirbyYoshiTongue = 0x9D,

    UnknownProjectile = 0xFF,
}

impl ProjectileType {
    pub fn from_u16(val: u16) -> Self {
        use ProjectileType::*;
        match val {
            0x06 => BobOmb,
            0x07 => MrSaturn,
            0x0C => Beamsword,
            0x30 => MarioFireball,
            0x31 => DrMarioCapsule,
            0x32 => KirbyCutter,
            0x33 => KirbyHammer,
            0x36 => FoxLaser,
            0x37 => FalcoLaser,
            0x38 => FoxShadow,
            0x39 => FalcoShadow,
            0x3A => LinkBomb,
            0x3B => YlinkBomb,
            0x3C => LinkBoomerang,
            0x3D => YlinkBoomerang,
            0x3E => LinkHookshot,
            0x3F => YlinkHookshot,
            0x40 => Arrow,
            0x41 => FireArrow,
            0x42 => PkFire,
            0x43 => PkFlash1,
            0x44 => PkFlash2,
            0x45 => PkThunderHead,
            0x46 => PkThunderTail1,
            0x47 => PkThunderTail2,
            0x48 => PkThunderTail3,
            0x49 => PkThunderTail4,
            0x4C => LinkArrow,
            0x4D => YlinkArrow,
            0x4E => PkFlashExplosion,
            0x4F => NeedleThrown,
            0x51 => PikachuThunder,
            0x52 => PichuThunder,
            0x53 => MarioCape,
            0x54 => DrMarioCape,
            0x55 => SheikSmoke,
            0x56 => YoshiEggThrown,
            0x57 => YoshiTongue,
            0x58 => YoshiStar,
            0x59 => PikachuThunderjolt1,
            0x5A => PikachuThunderjolt2,
            0x5B => PichuThunderjolt1,
            0x5C => PichuThunderjolt2,
            0x5D => SamusBomb,
            0x5E => SamusChargeBeam,
            0x5F => SamusMissle,
            0x60 => SamusGrappleBeam,
            0x61 => SheikChain,
            0x63 => Turnip,
            0x64 => BowserFlame,
            0x65 => NessBatt,
            0x66 => NessYoyo,
            0x67 => PeachParasol,
            0x69 => LuigiFire,
            0x6A => IceBlock,
            0x6B => IcBlizzard,
            0x6C => ZeldaFire,
            0x6D => ZeldaFireExplosion,
            0x6E => MewtoDisable,
            0x6F => ToadSpore,
            0x70 => Shadowball,
            0x71 => IcUpB,
            0x72 => Pesticide,
            0x73 => Manhole,
            0x74 => GwFire,
            0x75 => Parachute,
            0x76 => Turtle,
            0x77 => Sperky,
            0x78 => Judge,
            0x7A => Sausage,
            0x7B => YlinkMilk,
            0x7C => Firefighter,
            0x82 => KirbyMarioFire,
            0x83 => KirbyDrMarioFire,
            0x84 => KirbyLuigiFire,
            0x85 => KirbyIcBlock,
            0x87 => KirbyToadSpore,
            0x88 => KirbyFoxLaser,
            0x89 => KirbyFalcoLaser,
            0x8C => KirbyLinkArrow,
            0x8D => KirbyYlinkArrow,
            0x8E => KirbyLinkArrow2,
            0x8F => KirbyYlinkArrow2,
            0x90 => KirbyShadowball,
            0x91 => KirbyPkFlash,
            0x92 => KirbyPkFlashExplosion,
            0x93 => KirbyPikachuThunderjolt1,
            0x94 => KirbyPikachuThunderjolt2,
            0x95 => KirbyPichuThunderjolt1,
            0x96 => KirbyPichuThunderjolt2,
            0x97 => KirbySamusChargeshot,
            0x98 => KirbySheikNeedleThrown,
            0x99 => KirbySheikNeedleGround,
            0x9A => KirbyBowserFlame,
            0x9B => KirbySausage,
            0x9D => KirbyYoshiTongue,
            _ => UnknownProjectile,
        }
    }
}

impl Default for ProjectileType {
    fn default() -> Self {
        ProjectileType::UnknownProjectile
    }
}

// =====================
// Python helpers (module-level parity with Python module functions)
// =====================

#[pyfunction]
fn to_internal_stage(stage_id: u16) -> Stage {
    Stage::to_internal_stage(stage_id)
}

#[pyfunction]
fn to_internal(char_id: u8) -> Character {
    Character::from_css_id(char_id)
}

#[pyfunction]
fn from_internal(character: Character) -> u8 {
    character.to_css_id()
}

// Note: Enums are registered in lib.rs via the main melee_rust module