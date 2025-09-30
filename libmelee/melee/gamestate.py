""" Gamestate is a single snapshot in time of the game that represents all necessary information
        to make gameplay decisions
"""
from dataclasses import dataclass, field

import numpy as np

from libmelee.melee import enums


@dataclass
class Position:
    """Dataclass for position types. Has (x, y) coords."""
    x: np.float32 = np.float32(0)
    y: np.float32 = np.float32(0)

Speed = Position
Cursor = Position

@dataclass
class ECB:
    """ECBs (Environmental collision box) info. It's a diamond with four points that define it."""
    top: Position = field(default_factory=Position)
    bottom: Position = field(default_factory=Position)
    left: Position = field(default_factory=Position)
    right: Position = field(default_factory=Position)

class GameState(object):
    """Represents the state of a running game of Melee at a given moment in time"""
    __slots__ = ('frame', 'stage', 'menu_state', 'submenu', 'player', 'players', 'projectiles',
                 'ready_to_start', 'distance', 'menu_selection', '_newframe', 'playedOn', 'startAt',
                 'consoleNick', 'is_teams', 'custom', 'stage_select_cursor_x', 'stage_select_cursor_y')
    def __init__(self):
        self.frame = -10000
        """int: The current frame number. Monotonically increases. Can be negative."""
        self.stage = enums.Stage.FINAL_DESTINATION
        """enums.Stage: The current stage being played on"""
        self.menu_state = enums.Menu.IN_GAME
        """enums.MenuState: The current menu scene, such as IN_GAME, or STAGE_SELECT"""
        self.submenu = enums.SubMenu.UNKNOWN_SUBMENU
        """(enums.SubMenu): The current sub-menu"""
        self.players: dict[int, PlayerState] = dict()
        """(dict of int - gamestate.PlayerState): Dict of PlayerState objects. Key is controller port"""
        self.player = self.players
        """(dict of int - gamestate.PlayerState): WARNING: Deprecated. Will be removed in version 1.0.0. Use `players` instead
                Dict of PlayerState objects. Key is controller port"""
        self.projectiles = []
        """(list of Projectile): All projectiles (items) currently existing"""
        self.ready_to_start = False
        """(bool): Is the 'ready to start' banner showing at the character select screen?"""
        self.is_teams = False
        """(bool): Is this a teams game?"""
        self.distance = 0.0
        """(float): Euclidian distance between the two players. (or just Popo for climbers)"""
        self.menu_selection = 0
        """(int): The index of the selected menu item for when in menus."""
        self._newframe = True
        self.custom = dict()
        """(dict): Custom fields to be added by the user"""

        self.stage_select_cursor_x = 0.0
        self.stage_select_cursor_y = 0.0

class PlayerState(object):
    """ Represents the state of a single player """
    __slots__ = ('character', 'character_selected', 'percent', 'shield_strength', 'stock', 'facing',
                 'action', 'action_frame', 'invulnerable', 'invulnerability_left', 'hitlag_left', 'hitstun_frames_left',
                 'jumps_left', 'on_ground', 'speed_air_x_self', 'speed_y_self', 'speed_x_attack', 'speed_y_attack',
                 'speed_ground_x_self', 'controller_status', 'off_stage', 'iasa',
                 'controller_state', 'ecb_bottom', 'ecb_top', 'ecb_left', 'ecb_right',
                'nana', 'position', 'ecb', 'nickName', 'connectCode',
                 'displayName', 'team_id', 'is_powershield',
                 'is_reflect_active', 'is_subaction_invulnerable', 'is_fastfalling', 'is_defender_in_hitlag',
                 'is_in_hitlag', 'is_holding_character', 'is_shield_active', 'is_in_hitstun',
                 'is_touching_shield', 'is_cloaked', 'is_follower', 'is_inactive', 'is_dead', 'is_offscreen',
                 'l_cancel_status', 'cursor_x', 'cursor_y', 'cursor', 'coin_down', 'is_holding_cpu_slider', 'cpu_level')
    def __init__(self):
        # This value is what the character currently is IN GAME
        #   So this will have no meaning while in menus
        #   Also, this will change dynamically if you change characters
        #       IE: Shiek/Zelda
        self.character = enums.Character.UNKNOWN_CHARACTER
        """(enum.Character): The player's current character"""
        # This value is what character is selected at the character select screen
        #   Don't use this value when in-game
        self.character_selected = enums.Character.UNKNOWN_CHARACTER
        self.position = Position()
        """(Position): x, y character position"""
        self.percent = 0
        """(int): The player's damage"""
        self.shield_strength = 60.
        """(float): The player's shield strength (max 60). Shield breaks at 0"""
        self.is_powershield = False
        """(bool): Is the current action a Powershield? (not directly determinable via action states)"""
        self.stock = 0
        """(int): The player's remaining stock count"""
        self.facing = True
        """(bool): Is the character facing right? (left is False). Characters in Melee must always be facing left or right"""
        self.action = enums.Action.UNKNOWN_ANIMATION
        """(enum.Action): The current action (or animation) the character is in"""
        self.action_frame = 0
        """(int): What frame of the Action is the character in? Indexed from 1."""
        self.invulnerable = False
        """(bool): Is the player invulnerable?"""
        self.invulnerability_left = 0
        """(int): How many frames of invulnerability are left."""
        self.hitlag_left = 0
        """(bool): How many more frames of hitlag there is"""
        self.hitstun_frames_left = 0
        """(int): How many more frames of hitstun there is"""
        self.jumps_left = 0
        """(int): Number of jumps available. Including ground jump. Will be 2 for most characters on ground."""
        self.on_ground = True
        """(bool): Is the character on the ground?"""
        self.speed_air_x_self = 0
        """(float): Self-induced horizontal air speed"""
        self.speed_y_self = 0
        """(float): Self-induced vertical speed"""
        self.speed_x_attack = 0
        """(float): Attack-induced horizontal speed"""
        self.speed_y_attack = 0
        """(float): Attack-induced vertical speed"""
        self.speed_ground_x_self = 0
        """(float): Self-induced horizontal ground speed"""
        self.nana = None
        """(enums.PlayerState): Additional player state for Nana, if applicable.
                If the character is not Ice Climbers, Nana will be None.
                Will also be None if this player state is Nana itself.
                Lastly, the secondary climber is called 'Nana' here, regardless of the costume used."""
        self.controller_status = enums.ControllerStatus.CONTROLLER_UNPLUGGED
        """(enums.ControllerStatus): Status of the player's controller."""
        self.off_stage = False
        """(bool): Helper variable to say if the character is 'off stage'. """
        self.iasa = 0
        from libmelee.melee.controller import ControllerState
        self.controller_state = ControllerState()
        """(controller.ControllerState): What buttons were pressed for this character"""
        self.ecb = ECB()
        self.ecb_right = (0, 0)
        """(float, float): Right edge of the ECB. (x, y) offset from player's center."""
        self.ecb_left = (0, 0)
        """(float, float): Left edge of the ECB. (x, y) offset from player's center."""
        self.ecb_top = (0, 0)
        """(float, float): Top edge of the ECB. (x, y) offset from player's center."""
        self.ecb_bottom = (0, 0)
        """(float, float): Bottom edge of the ECB. (x, y) offset from player's center."""
        self.is_reflect_active = False
        """(bool): Is reflect active"""
        self.is_subaction_invulnerable = False
        """(bool): Has temporary intangibility or invincibility from subaction"""
        self.is_fastfalling = False
        """(bool): Is fastfalling"""
        self.is_defender_in_hitlag = False
        """(bool): Is the defender in hitlag (not shield hitlag)"""
        self.is_in_hitlag = False
        """(bool): Is in hitlag"""
        self.is_holding_character = False
        """(bool): Is holding another character due to grab/command grab"""
        self.is_shield_active = False
        """(bool): Is shield active"""
        self.is_in_hitstun = False
        """(bool): Is in hitstun"""
        self.is_dead = False
        """(bool): Is dead"""
        self.is_offscreen = False
        """(bool): Is offscreen"""
        self.l_cancel_status: int = 0
        """(int): 0 = none, 1 = successful, 2 = unsuccessful"""

        self.cursor = Cursor()
        """(Position): x, y cursor position"""
        self.cursor_x = 0
        """(float): DEPRECATED. Use `cursor` instead. Will be removed in 1.0.0. Cursor X value"""
        self.cursor_y = 0
        """(float): DEPRECATED. Use `position` instead. Will be removed in 1.0.0. Cursor Y value"""
        self.coin_down = False
        """(bool): Is the player's character selection coin placed down? (Does not work in Slippi selection screen)"""
        self.is_holding_cpu_slider = False
        """(bool): Is the player holding the CPU slider in the character select screen?"""
        self.cpu_level = 0

class Projectile:
    """ Represents the state of a projectile (items, lasers, etc...) """
    def __init__(self):
        self.position = Position()
        """(Position): x, y projectile position"""
        self.x = 0
        """(float): DEPRECATED. Use `position` instead. Will be removed in 1.0.0. Projectile's X position"""
        self.y = 0
        """(float): DEPRECATED. Use `position` instead. Will be removed in 1.0.0. Projectile's Y position"""
        self.speed = Speed()
        """(Position): x, y projectile speed"""
        self.x_speed = 0
        """(float): DEPRECATED. Use `speed` instead. Will be removed in 1.0.0. Projectile's horizontal speed"""
        self.y_speed = 0
        """(float): DEPRECATED. Use `speed` instead. Will be removed in 1.0.0. Projectile's vertical speed"""
        self.owner = -1
        """(int): Player port of the projectile's owner. -1 for no owner"""
        self.type = enums.ProjectileType.UNKNOWN_PROJECTILE
        """(enums.ProjectileType): Which actual projectile type this is"""
        self.frame = 0
        """(int): How long the item has been out"""
        self.subtype = 0
        """(int): The subtype of the item. Many projectiles have 'subtypes' that make them different. They're all different, so it's not an enum"""

def port_detector(gamestate, character, costume):
    """Autodiscover what port the given character is on

    Slippi Online assigns us a random port when playing online. Find out which we are

    Returns:
        [1-4]: The given character belongs to the returned port
        0: We don't know.

    Args:
        gamestate: Current gamestate
        character: The character we know we picked
        costume: Costume index we picked
    """
    detected_port = 0
    for i, player in gamestate.players.items():
        if player.character == character and player.costume == costume:
            if detected_port > 0:
                return 0
            detected_port = i

    return detected_port
