use pyo3::prelude::*;
use byteorder::{BigEndian, ByteOrder};
use crate::enums::{Action, Character, Menu, ProjectileType, Stage};
use crate::gamestate::{GameState, Projectile};

// Event type constants
const EVENT_GECKO_CODES: u8 = 0x10;
const EVENT_PAYLOADS: u8 = 0x35;
const EVENT_GAME_START: u8 = 0x36;
const EVENT_PRE_FRAME: u8 = 0x37;
const EVENT_POST_FRAME: u8 = 0x38;
const EVENT_GAME_END: u8 = 0x39;
const EVENT_FRAME_START: u8 = 0x3A;
const EVENT_ITEM_UPDATE: u8 = 0x3B;
const EVENT_FRAME_BOOKEND: u8 = 0x3C;
const EVENT_MENU_EVENT: u8 = 0x3E;

#[pyclass]
pub struct SlpParser {
    event_sizes: [usize; 256],
    current_stage: Stage,
    frame_num: i32,
    is_teams: bool,
    costumes: [u8; 4],
    cpu_level: [u8; 4],
    team_id: [u8; 4],
    last_frame: i32,  // Last frame number seen (for old files)
    has_bookends: bool,  // Whether this file has FRAME_BOOKEND events
    finished: bool,  // Whether we've reached the end (to avoid returning last frame multiple times)
    debug: bool,  // Debug logging enabled via LIBMELEE_DEBUG env var
    buffered_bytes: Vec<u8>,  // Bytes we need to revisit on next call when emulating manual bookends
}

#[pymethods]
impl SlpParser {
    #[new]
    pub fn new() -> Self {
        let debug = std::env::var("LIBMELEE_DEBUG")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        
        if debug {
            eprintln!("[RUST DEBUG] SlpParser created - debug logging ENABLED");
        }
        
        SlpParser {
            event_sizes: [0; 256],
            current_stage: Stage::NoStage,
            frame_num: -10000,
            is_teams: false,
            costumes: [0; 4],
            cpu_level: [0; 4],
            team_id: [0; 4],
            last_frame: -10000,
            has_bookends: false,
            finished: false,
            debug,
            buffered_bytes: Vec::new(),
        }
    }
    
    pub fn parse_events(&mut self, event_bytes: &[u8], gamestate: &mut GameState) -> PyResult<bool> {
        self.parse_events_impl(event_bytes, gamestate)
    }
}

impl SlpParser {
    fn parse_events_impl(&mut self, event_bytes: &[u8], gamestate: &mut GameState) -> PyResult<bool> {
        let owned_buffer = if !self.buffered_bytes.is_empty() {
            let mut combined = std::mem::take(&mut self.buffered_bytes);
            if !event_bytes.is_empty() {
                combined.extend_from_slice(event_bytes);
            }
            Some(combined)
        } else {
            None
        };
        let input_bytes: &[u8] = if let Some(ref buf) = owned_buffer {
            buf.as_slice()
        } else {
            event_bytes
        };

        if self.debug {
            eprintln!("[RUST DEBUG] === parse_events called ===");
            eprintln!("[RUST DEBUG] Buffer size: {} bytes", input_bytes.len());
            eprintln!("[RUST DEBUG] State: has_bookends={}, last_frame={}, frame_num={}, finished={}", 
                     self.has_bookends, self.last_frame, self.frame_num, self.finished);
        }

        gamestate.menu_state = Menu::InGame;
        let mut offset = 0;

        while offset < input_bytes.len() {
            let command_byte = input_bytes[offset];
            
            // Handle PAYLOADS event
            if command_byte == EVENT_PAYLOADS {
                if self.debug {
                    eprintln!("[RUST DEBUG] [Offset {}] EVENT_PAYLOADS (0x{:02x})", offset, command_byte);
                }
                
                if offset + 1 >= input_bytes.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("PAYLOADS event at offset {} missing size byte", offset)
                    ));
                }
                let payload_size = input_bytes[offset + 1] as usize;
                let mut cursor = offset + 2;
                let num_commands = (payload_size - 1) / 3;
                
                if self.debug {
                    eprintln!("[RUST DEBUG]   Payload size: {}, num_commands: {}", payload_size, num_commands);
                }
                
                for _ in 0..num_commands {
                    if cursor + 3 > input_bytes.len() {
                        break;
                    }
                    let cmd = input_bytes[cursor];
                    let cmd_len = BigEndian::read_u16(&input_bytes[cursor + 1..cursor + 3]) as usize;
                    self.event_sizes[cmd as usize] = cmd_len + 1;
                    
                    if self.debug {
                        eprintln!("[RUST DEBUG]     Event 0x{:02x} -> size {}", cmd, cmd_len + 1);
                    }
                    
                    cursor += 3;
                }
                
                // Check if this file has FRAME_BOOKEND events
                self.has_bookends = self.event_sizes[EVENT_FRAME_BOOKEND as usize] > 0;
                
                if self.debug {
                    eprintln!("[RUST DEBUG]   has_bookends = {} (FRAME_BOOKEND size = {})", 
                             self.has_bookends, self.event_sizes[EVENT_FRAME_BOOKEND as usize]);
                }
                
                offset += payload_size + 1;
                continue;
            }
            
            let event_size = self.event_sizes[command_byte as usize];
            if event_size == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown event 0x{:02x} at offset {}. Event size not set - likely unsupported Slippi version or corrupted file.",
                            command_byte, offset)
                ));
            }
                if offset + event_size > input_bytes.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Event 0x{:02x} size {} would exceed buffer at offset {} (buffer len={}). File may be corrupted.",
                            command_byte, event_size, offset, input_bytes.len())
                ));
            }

            let chunk = &input_bytes[offset..offset + event_size];
            
            if self.debug {
                eprintln!("[RUST DEBUG] [Offset {}] Event 0x{:02x}, size {}", offset, command_byte, event_size);
            }
            
            offset += event_size;
            
            match command_byte {
                EVENT_FRAME_START => {
                    if self.debug {
                        eprintln!("[RUST DEBUG]   -> FRAME_START (ignoring)");
                    }
                    continue;
                }
                EVENT_GECKO_CODES => {
                    if self.debug {
                        eprintln!("[RUST DEBUG]   -> GECKO_CODES (ignoring)");
                    }
                    continue;
                }
                EVENT_GAME_START => {
                    if self.debug {
                        eprintln!("[RUST DEBUG]   -> GAME_START");
                    }
                    self.parse_game_start(chunk, gamestate);
                }
                EVENT_GAME_END => {
                    if self.debug {
                        eprintln!("[RUST DEBUG]   -> GAME_END");
                    }
                    if !self.has_bookends && !self.finished {
                        if self.debug {
                            eprintln!("[RUST DEBUG]     Manual bookends - flushing final frame on GAME_END");
                        }
                        self.finished = true;
                        return Ok(true);
                    }
                    continue;
                }
                EVENT_PRE_FRAME => {
                    if self.debug {
                        eprintln!("[RUST DEBUG]   -> PRE_FRAME");
                    }
                    
                    // Check for frame boundary BEFORE parsing (ONLY for old files without bookends)
                    if !self.has_bookends && chunk.len() >= 5 {
                        let frame = BigEndian::read_i32(&chunk[1..5]);
                        
                        if self.debug {
                            eprintln!("[RUST DEBUG]     Frame boundary check: frame={}, last_frame={}", frame, self.last_frame);
                        }
                        
                        // Initialize last_frame on first frame
                        if self.last_frame < -123 {
                            if self.debug {
                                eprintln!("[RUST DEBUG]     First frame detected: {}", frame);
                            }
                            self.last_frame = frame;
                        } else if frame > self.last_frame {
                            // New frame detected! Update last_frame BEFORE returning
                            // so subsequent events for this frame don't think it's new again
                            if self.debug {
                                eprintln!("[RUST DEBUG]     *** NEW FRAME DETECTED *** (prev={}, new={}) - RETURNING TRUE", self.last_frame, frame);
                            }
                            self.last_frame = frame;
                            let rewind_start = offset - event_size;
                            self.buffered_bytes = input_bytes[rewind_start..].to_vec();
                            return Ok(true);
                        }
                        // Update last_frame to current frame
                        self.last_frame = frame;
                    }
                    
                    self.parse_pre_frame(chunk, gamestate)?;
                }
                EVENT_POST_FRAME => {
                    if chunk.len() >= 5 {
                        let frame = BigEndian::read_i32(&chunk[1..5]);
                        if self.debug {
                            eprintln!("[RUST DEBUG]   -> POST_FRAME (frame={})", frame);
                        }
                    } else if self.debug {
                        eprintln!("[RUST DEBUG]   -> POST_FRAME");
                    }
                    
                    // Check for frame boundary BEFORE parsing (ONLY for old files without bookends)
                    if !self.has_bookends && chunk.len() >= 5 {
                        let frame = BigEndian::read_i32(&chunk[1..5]);
                        
                        if self.debug {
                            eprintln!("[RUST DEBUG]     Frame boundary check: frame={}, last_frame={}", frame, self.last_frame);
                        }
                        
                        // Initialize last_frame on first frame
                        if self.last_frame < -123 {
                            if self.debug {
                                eprintln!("[RUST DEBUG]     First frame detected: {}", frame);
                            }
                            self.last_frame = frame;
                        } else if frame > self.last_frame {
                            // New frame detected! Update last_frame BEFORE returning
                            // so subsequent events for this frame don't think it's new again
                            if self.debug {
                                eprintln!("[RUST DEBUG]     *** NEW FRAME DETECTED *** (prev={}, new={}) - RETURNING TRUE", self.last_frame, frame);
                            }
                            self.last_frame = frame;
                            let rewind_start = offset - event_size;
                            self.buffered_bytes = input_bytes[rewind_start..].to_vec();
                            return Ok(true);
                        }
                        // Update last_frame to current frame
                        self.last_frame = frame;
                    }
                    
                    self.parse_post_frame(chunk, gamestate);
                }
                EVENT_FRAME_BOOKEND => {
                    if self.debug {
                        eprintln!("[RUST DEBUG]   -> FRAME_BOOKEND");
                    }
                    
                    self.parse_frame_bookend(gamestate);
                    
                    if self.debug {
                        eprintln!("[RUST DEBUG]     gamestate.frame={}, frame_num={}", gamestate.frame, self.frame_num);
                    }
                    
                    if gamestate.frame <= self.frame_num {
                        if self.debug {
                            eprintln!("[RUST DEBUG]     Rollback/duplicate frame detected - RETURNING FALSE");
                        }
                        return Ok(false);  // Skip rollback frames
                    }
                    self.frame_num = gamestate.frame;
                    
                    if self.debug {
                        eprintln!("[RUST DEBUG]     Frame complete - RETURNING TRUE");
                    }
                    return Ok(true);
                }
                EVENT_ITEM_UPDATE => {
                    if self.debug {
                        eprintln!("[RUST DEBUG]   -> ITEM_UPDATE");
                    }
                    self.parse_item_update(chunk, gamestate);
                }
                EVENT_MENU_EVENT => {
                    eprintln!("WARNING: Got menu event in game stream");
                    continue;
                }
                _ => {
                    eprintln!("WARNING: Unhandled event type 0x{:02x}", command_byte);
                }
            }
        }
        
        if self.debug {
            eprintln!("[RUST DEBUG] End of event stream reached");
            eprintln!("[RUST DEBUG] has_bookends={}, last_frame={}, finished={}", 
                     self.has_bookends, self.last_frame, self.finished);
            eprintln!("[RUST DEBUG] No more frames to return - RETURNING FALSE");
        }

        Ok(false)
    }
    
    fn parse_game_start(&mut self, data: &[u8], _gamestate: &mut GameState) {
        // Set initial frame number based on file format
        self.frame_num = -10000;
        self.last_frame = -10000;
        self.finished = false;
        self.buffered_bytes.clear();

        if data.len() >= 0x15 {
            let stage_id = BigEndian::read_u16(&data[0x13..0x15]);
            self.current_stage = Stage::to_internal_stage(stage_id);
        }
        
        if data.len() >= 0x0F {
            self.is_teams = BigEndian::read_u16(&data[0x0D..0x0F]) != 0;
        }
        
        // Parse costumes
        for i in 0..4 {
            let offset = 0x68 + (0x24 * i);
            if data.len() > offset {
                self.costumes[i] = data[offset];
            }
        }
        
        // Parse CPU levels
        for i in 0..4 {
            let offset = 0x74 + (0x24 * i);
            if data.len() > offset {
                self.cpu_level[i] = data[offset];
            }
        }
        
        // Parse team IDs
        for i in 0..4 {
            let offset = 0x6E + (0x24 * i);
            if data.len() > offset {
                self.team_id[i] = data[offset];
            }
        }
        
        // Clear CPU levels for non-CPU players
        for i in 0..4 {
            let offset = 0x66 + (0x24 * i);
            if data.len() > offset && data[offset] != 1 {
                self.cpu_level[i] = 0;
            }
        }
    }
    
    #[inline(always)]
    fn parse_pre_frame(&mut self, data: &[u8], gamestate: &mut GameState) -> PyResult<()> {
        if data.len() < 0x33 {
            return Ok(());
        }
        
        let controller_port = (data[0x5] + 1) as u8;
        let is_nana = data[0x6] == 1;
        
        let player = gamestate.ensure_player(controller_port);
        
        // Handle Nana
        let target_player = if is_nana {
            player.ensure_nana()
        } else {
            player
        };
        
        // Parse stick values (4 floats starting at 0x19)
        if data.len() >= 0x29 {
            let main_x = BigEndian::read_f32(&data[0x19..0x1D]);
            let main_y = BigEndian::read_f32(&data[0x1D..0x21]);
            let c_x = BigEndian::read_f32(&data[0x21..0x25]);
            let c_y = BigEndian::read_f32(&data[0x25..0x29]);
            
            target_player.controller_state.main_stick = (
                (main_x / 2.0) + 0.5,
                (main_y / 2.0) + 0.5,
            );
            target_player.controller_state.c_stick = (
                (c_x / 2.0) + 0.5,
                (c_y / 2.0) + 0.5,
            );
        }
        
        // Parse raw main stick
        if data.len() > 0x3B {
            let raw_x = data[0x3B] as i8;
            target_player.controller_state.raw_main_stick.0 = raw_x;
        }
        if data.len() > 0x40 {
            let raw_y = data[0x40] as i8;
            target_player.controller_state.raw_main_stick.1 = raw_y;
        }
        
        // Parse trigger
        if data.len() >= 0x2D {
            let trigger = BigEndian::read_f32(&data[0x29..0x2D]);
            target_player.controller_state.l_shoulder = trigger;
            target_player.controller_state.r_shoulder = trigger;
        }
        
        // Parse button bits
        if data.len() >= 0x33 {
            let button_bits = BigEndian::read_u16(&data[0x31..0x33]);
            target_player.controller_state.set_button_flag("BUTTON_A", button_bits & 0x0100 != 0)?;
            target_player.controller_state.set_button_flag("BUTTON_B", button_bits & 0x0200 != 0)?;
            target_player.controller_state.set_button_flag("BUTTON_X", button_bits & 0x0400 != 0)?;
            target_player.controller_state.set_button_flag("BUTTON_Y", button_bits & 0x0800 != 0)?;
            target_player.controller_state.set_button_flag("BUTTON_START", button_bits & 0x1000 != 0)?;
            target_player.controller_state.set_button_flag("BUTTON_Z", button_bits & 0x0010 != 0)?;
            target_player.controller_state.set_button_flag("BUTTON_R", button_bits & 0x0020 != 0)?;
            target_player.controller_state.set_button_flag("BUTTON_L", button_bits & 0x0040 != 0)?;
        }

        Ok(())
    }
    
    #[inline(always)]
    fn parse_post_frame(&mut self, data: &[u8], gamestate: &mut GameState) {
        if data.len() < 0x22 {
            return;
        }
        
        gamestate.stage = self.current_stage;
        gamestate.is_teams = self.is_teams;
        let frame_from_data = BigEndian::read_i32(&data[0x1..0x5]);
        if self.debug {
            eprintln!("[RUST DEBUG]   Setting gamestate.frame: {} -> {}", gamestate.frame, frame_from_data);
        }
        gamestate.frame = frame_from_data;
        
        let controller_port = (data[0x5] + 1) as u8;
        let is_nana = data[0x6] == 1;
        
        let player = gamestate.ensure_player(controller_port);
        
        // Handle Nana
        let target_player = if is_nana {
            player.ensure_nana()
        } else {
            player
        };
        
        // Position
        if data.len() >= 0x12 {
            target_player.position.x = BigEndian::read_f32(&data[0x0A..0x0E]);
            target_player.position.y = BigEndian::read_f32(&data[0x0E..0x12]);
        }
        
        // Character
        if data.len() > 0x7 {
            target_player.character = Character::from_u8(data[0x7]);
        }
        
        // Action
        if data.len() >= 0x0A {
            let action_val = BigEndian::read_u16(&data[0x8..0x0A]);
            target_player.action = Action::from_u16(action_val);
        }
        
        // Facing
        if data.len() >= 0x16 {
            let facing_val = BigEndian::read_f32(&data[0x12..0x16]);
            target_player.facing = facing_val > 0.0;
        }
        
        // Percent and shield
        if data.len() >= 0x1E {
            let percent = BigEndian::read_f32(&data[0x16..0x1A]);
            target_player.percent = percent as i32;
            target_player.shield_strength = BigEndian::read_f32(&data[0x1A..0x1E]);
        }
        
        // Stock
        if data.len() > 0x21 {
            target_player.stock = data[0x21];
        }
        
        // Action frame
        if data.len() >= 0x26 {
            let action_frame = BigEndian::read_f32(&data[0x22..0x26]);
            target_player.action_frame = action_frame as i32;
        }
        
        // Status bytes
        if data.len() > 0x2A {
            let sb1 = data[0x26];
            let sb2 = data[0x27];
            let sb3 = data[0x28];
            let sb4 = data[0x29];
            let sb5 = data[0x2A];
            
            target_player.is_reflect_active = (sb1 & 0x10) != 0;
            target_player.is_subaction_invulnerable = (sb2 & 0x04) != 0;
            target_player.is_fastfalling = (sb2 & 0x08) != 0;
            target_player.is_defender_in_hitlag = (sb2 & 0x10) != 0;
            target_player.is_in_hitlag = (sb2 & 0x20) != 0;
            target_player.is_holding_character = (sb3 & 0x04) != 0;
            target_player.is_shield_active = (sb3 & 0x80) != 0;
            target_player.is_in_hitstun = (sb4 & 0x02) != 0;
            target_player.is_powershield = (sb4 & 0x20) != 0;
            target_player.is_dead = (sb5 & 0x40) != 0;
            target_player.is_offscreen = (sb5 & 0x80) != 0;
        }
        
        // Hitstun frames left
        if data.len() >= 0x2F {
            let hitstun = BigEndian::read_f32(&data[0x2B..0x2F]);
            target_player.hitstun_frames_left = hitstun as i32;
        }
        
        // On ground
        if data.len() > 0x2F {
            target_player.on_ground = data[0x2F] == 0;
        }
        
        // Jumps left
        if data.len() > 0x32 {
            target_player.jumps_left = data[0x32];
        }
        
        // L-cancel status
        if data.len() > 0x33 {
            target_player.l_cancel_status = data[0x33];
        }
        
        // Invulnerable
        if data.len() > 0x34 {
            target_player.invulnerable = data[0x34] != 0;
        }
        
        // Speed values
        if data.len() >= 0x4D {
            target_player.speed_air_x_self = BigEndian::read_f32(&data[0x35..0x39]);
            target_player.speed_y_self = BigEndian::read_f32(&data[0x39..0x3D]);
            target_player.speed_x_attack = BigEndian::read_f32(&data[0x3D..0x41]);
            target_player.speed_y_attack = BigEndian::read_f32(&data[0x41..0x45]);
            target_player.speed_ground_x_self = BigEndian::read_f32(&data[0x45..0x49]);
            
            let hitlag = BigEndian::read_f32(&data[0x49..0x4D]);
            target_player.hitlag_left = hitlag as i32;
        }
        
        // ECB (Environmental Collision Box)
        if data.len() >= 0x6D {
            let ecb_top_x = BigEndian::read_f32(&data[0x4D..0x51]);
            let ecb_top_y = BigEndian::read_f32(&data[0x51..0x55]);
            let ecb_bottom_x = BigEndian::read_f32(&data[0x55..0x59]);
            let ecb_bottom_y = BigEndian::read_f32(&data[0x59..0x5D]);
            let ecb_left_x = BigEndian::read_f32(&data[0x5D..0x61]);
            let ecb_left_y = BigEndian::read_f32(&data[0x61..0x65]);
            let ecb_right_x = BigEndian::read_f32(&data[0x65..0x69]);
            let ecb_right_y = BigEndian::read_f32(&data[0x69..0x6D]);
            
            target_player.ecb.top.x = ecb_top_x;
            target_player.ecb.top.y = ecb_top_y;
            target_player.ecb.bottom.x = ecb_bottom_x;
            target_player.ecb.bottom.y = ecb_bottom_y;
            target_player.ecb.left.x = ecb_left_x;
            target_player.ecb.left.y = ecb_left_y;
            target_player.ecb.right.x = ecb_right_x;
            target_player.ecb.right.y = ecb_right_y;
            
            target_player.ecb_top = (ecb_top_x, ecb_top_y);
            target_player.ecb_bottom = (ecb_bottom_x, ecb_bottom_y);
            target_player.ecb_left = (ecb_left_x, ecb_left_y);
            target_player.ecb_right = (ecb_right_x, ecb_right_y);
        }
        
        // Calculate off_stage
        // Note: This requires stage edge positions, simplified for now
        target_player.off_stage = false;
    }
    
    fn parse_item_update(&mut self, data: &[u8], gamestate: &mut GameState) {
        if data.len() < 0x2B {
            return;
        }
        
        let mut projectile = Projectile::new();
        
        if data.len() >= 0x1C {
            projectile.position.x = BigEndian::read_f32(&data[0x14..0x18]);
            projectile.position.y = BigEndian::read_f32(&data[0x18..0x1C]);
            projectile.x = projectile.position.x;
            projectile.y = projectile.position.y;
        }
        
        if data.len() >= 0x14 {
            projectile.speed.x = BigEndian::read_f32(&data[0xC..0x10]);
            projectile.speed.y = BigEndian::read_f32(&data[0x10..0x14]);
            projectile.x_speed = projectile.speed.x;
            projectile.y_speed = projectile.speed.y;
        }
        
        if data.len() > 0x2A {
            let owner = data[0x2A] as i32 + 1;
            projectile.owner = if owner > 4 { -1 } else { owner };
        }
        
        if data.len() >= 0x7 {
            let type_val = BigEndian::read_u16(&data[0x5..0x7]);
            projectile.proj_type = ProjectileType::from_u16(type_val);
        }
        
        if data.len() >= 0x22 {
            let frame_val = BigEndian::read_f32(&data[0x1E..0x22]);
            projectile.frame = frame_val as i32;
        }
        
        if data.len() > 0x7 {
            projectile.subtype = data[0x7];
        }
        
        // Filter out certain projectiles
        if matches!(projectile.proj_type, ProjectileType::SamusBomb) && projectile.subtype == 3 {
            return;
        }
        if matches!(projectile.proj_type, ProjectileType::SamusMissle) && (projectile.subtype == 2 || projectile.subtype == 3) {
            return;
        }
        if matches!(projectile.proj_type, ProjectileType::SamusChargeBeam) && projectile.subtype == 0 {
            return;
        }
        
        gamestate.projectiles.push(projectile);
    }
    
    fn parse_frame_bookend(&mut self, gamestate: &mut GameState) {
        // Calculate distance between players
        let player_positions: Vec<(f32, f32)> = gamestate.players()
            .values()
            .take(2)
            .map(|p| (p.position.x, p.position.y))
            .collect();
        
        if player_positions.len() >= 2 {
            let dx = player_positions[0].0 - player_positions[1].0;
            let dy = player_positions[0].1 - player_positions[1].1;
            gamestate.distance = (dx * dx + dy * dy).sqrt();
        }
    }
}
