import numpy as np

class PossessionTracker:
    def __init__(self, smoothing_window=5, confidence_threshold=0.6):
        self.possession_history = []
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.current_possession = None
        self.possession_start_frame = 0
        
    def update(self, possession_result: dict, frame_number: int) -> dict:
        """Update possession with temporal smoothing."""
        self.possession_history.append(possession_result)
        
        # Keep only recent history
        if len(self.possession_history) > self.smoothing_window:
            self.possession_history.pop(0)
        
        # Analyze recent history
        recent_teams = [p["team"] for p in self.possession_history[-self.smoothing_window:]]
        recent_confidences = [p["confidence"] for p in self.possession_history[-self.smoothing_window:]]
        
        # Count team possessions
        team_counts = {0: 0, 1: 0, None: 0}
        for team in recent_teams:
            team_counts[team] += 1
        
        # Determine stable possession
        max_count = max(team_counts.values())
        stable_teams = [team for team, count in team_counts.items() if count == max_count]
        
        if len(stable_teams) == 1 and max_count >= 3:  # At least 3 consistent frames
            stable_team = stable_teams[0]
            avg_confidence = np.mean([c for p, c in zip(recent_teams, recent_confidences) 
                                    if p == stable_team])
            
            if avg_confidence >= self.confidence_threshold:
                if self.current_possession != stable_team:
                    self.current_possession = stable_team
                    self.possession_start_frame = frame_number
                
                return {
                    "team": stable_team,
                    "confidence": avg_confidence,
                    "duration": frame_number - self.possession_start_frame,
                    "stable": True
                }
        
        # Return current possession if no stable change
        return {
            "team": self.current_possession,
            "confidence": 0.5,
            "duration": frame_number - self.possession_start_frame if self.current_possession else 0,
            "stable": False
        }