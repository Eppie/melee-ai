import json
import os

file_path = './death_logs/death_0007_frame18473.json'

def analyze_death_log(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    frames = data.get('frames', [])
    total_frames = len(frames)
    print(f"Total frames recorded: {total_frames}")

    # Look at frames around the initiation (150-170)
    start_index = 150
    end_index = 170
    
    print(f"{'Frame':<6} | {'Action':<6} | {'Stick X':<7} | {'Stick Y':<7} | {'Btn B':<5} | {'Logit B':<8} | {'Prob B':<8}")
    print("-" * 70)

    import math

    def sigmoid(x):
      return 1 / (1 + math.exp(-x))

    for i in range(start_index, min(end_index, total_frames)):
        frame = frames[i]
        raw = frame.get('raw_features', {})
        logits = frame.get('logits', {})
        
        # Logits can be nested or flat, let's try to handle both based on the schema description
        # Schema: "logits": { "anyOf": [ { "buttons": [...] }, "null" ] }
        # The user said buttons are: ("button_a", "button_b", "button_xy", "button_z", "button_lr")
        # B button is at index 1
        
        b_logit_val = 0.0
        if logits and isinstance(logits, dict):
             buttons_logits = logits.get('buttons')
             if buttons_logits and len(buttons_logits) > 1:
                 b_logit_val = buttons_logits[1]
        
        p1_action = raw.get('p1_action')
        stick_x = raw.get('p1_main_stick_x')
        stick_y = raw.get('p1_main_stick_y')
        btn_b = raw.get('p1_button_b')

        prob_b = sigmoid(b_logit_val)

        print(f"{i:<6} | {p1_action:<6} | {stick_x:<7.2f} | {stick_y:<7.2f} | {btn_b:<5.2f} | {b_logit_val:<8.3f} | {prob_b:<8.3f}")

if __name__ == "__main__":
    analyze_death_log(file_path)
