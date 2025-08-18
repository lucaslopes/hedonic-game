class Game:
    """A class representing a hedonic game."""
    
    def __init__(self, name: str = "Default Game"):
        self.name = name
        self.players = []
    
    def add_player(self, player: str) -> None:
        """Add a player to the game."""
        self.players.append(player)
    
    def get_players(self) -> list[str]:
        """Get all players in the game."""
        return self.players.copy()
    
    def __str__(self) -> str:
        return f"Game: {self.name} with {len(self.players)} players"

# Make Game the main export
__all__ = ["Game"]
