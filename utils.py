import re
from typing import List, Tuple, Dict, Optional

class RegionEnforcer:
    def __init__(self) -> None:
        self.char_to_int: Dict[str, str] = {
            'O': '0', 'I': '1', 'Z': '2', 'B': '8', 'S': '5', 'A': '4', 'G': '6'
        }
        self.int_to_char: Dict[str, str] = {
            '0': 'O', '1': 'I', '2': 'Z', '8': 'B', '5': 'S', '4': 'A', '6': 'G'
        }

    def clean(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r'[^a-zA-Z0-9]', '', text).upper()

    def enforce_ua(self, text: str) -> str:
        clean_text = self.clean(text)
        
        best_candidate: Optional[str] = None
        max_score: int = -1

        if len(clean_text) < 8:
            return text 

        for i in range(len(clean_text) - 7):
            segment = list(clean_text[i : i+8])
            current_score = 0
            
            for idx in [0, 1, 6, 7]:
                if segment[idx].isalpha(): current_score += 1
            
            for idx in range(2, 6):
                if segment[idx].isdigit(): current_score += 1
            
            if current_score > max_score:
                max_score = current_score
                
                for idx in [0, 1, 6, 7]:
                    if segment[idx] in self.int_to_char:
                        segment[idx] = self.int_to_char[segment[idx]]
                
                for idx in range(2, 6):
                    if segment[idx] in self.char_to_int:
                        segment[idx] = self.char_to_int[segment[idx]]
                
                best_candidate = "".join(segment)

        if max_score >= 5 and best_candidate:
            return best_candidate
        return clean_text

    def enforce_eu_general(self, text: str) -> str:
        clean_text = self.clean(text)
        prefixes: List[str] = ['UA', 'PL', 'LT', 'D', 'CZ', 'SK', 'II', '11']
        
        for p in prefixes:
            if clean_text.startswith(p) and len(clean_text) > len(p) + 4:
                return clean_text[len(p):]
        return clean_text

enforcer = RegionEnforcer()

def fuzzy_check(
    detected_raw: str, 
    database_str: str, 
    country_mode: str = "Auto"
) -> Tuple[bool, str, str]:
    
    processed_plate: str = detected_raw
    debug_info: str = f"Mode: {country_mode}"

    if "Ukraine" in country_mode:
        processed_plate = enforcer.enforce_ua(detected_raw)
        debug_info += " -> UA Enforced"
    elif "Europe" in country_mode:
        processed_plate = enforcer.enforce_eu_general(detected_raw)
        debug_info += " -> EU Cleaned"
    else:
        processed_plate = enforcer.clean(detected_raw)
    
    db_list: List[str] = [enforcer.clean(plate) for plate in database_str.split(',')]
    
    if processed_plate in db_list:
        return True, processed_plate, debug_info
    
    for db_plate in db_list:
        if len(db_plate) == len(processed_plate):
            diff = sum(1 for a, b in zip(db_plate, processed_plate) if a != b)
            if diff <= 1:
                return True, db_plate, f"{debug_info} (Fuzzy fix)"

    return False, processed_plate, debug_info