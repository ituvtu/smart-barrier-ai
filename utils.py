import re

class RegionEnforcer:
    def __init__(self):
        # Карти замін (OCR часто плутає ці символи)
        self.char_to_int = {'O': '0', 'I': '1', 'Z': '2', 'B': '8', 'S': '5', 'A': '4', 'G': '6'}
        self.int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '8': 'B', '5': 'S', '4': 'A', '6': 'G'}

    def clean(self, text):
        """Базова чистка: тільки цифри та літери upper case"""
        return re.sub(r'[^a-zA-Z0-9]', '', text).upper()

    def enforce_ua(self, text):
        """
        Україна: 2 літери + 4 цифри + 2 літери (Всього 8 символів)
        Приклад: AA1234BB
        """
        clean_text = self.clean(text)
        
        # Метод "Ковзного вікна" (шукаємо найкращі 8 символів підряд)
        # Якщо OCR видав "VABO0001OO", ми перевіримо:
        # 1. "VABO0001" (Score низький)
        # 2. "ABO0001O" (Score низький)
        # 3. "BO0001OO" (Score високий!)
        
        best_candidate = None
        max_score = -1

        if len(clean_text) < 8:
            return text # Занадто короткий, повертаємо як є

        for i in range(len(clean_text) - 7):
            segment = list(clean_text[i : i+8])
            current_score = 0
            
            # --- ОЦІНКА ---
            # Перевіряємо структуру LL DDDD LL
            # Позиції 0,1 (Літери)
            if segment[0].isalpha(): current_score += 1
            if segment[1].isalpha(): current_score += 1
            # Позиції 2,3,4,5 (Цифри)
            if segment[2].isdigit(): current_score += 1
            if segment[3].isdigit(): current_score += 1
            if segment[4].isdigit(): current_score += 1
            if segment[5].isdigit(): current_score += 1
            # Позиції 6,7 (Літери)
            if segment[6].isalpha(): current_score += 1
            if segment[7].isalpha(): current_score += 1
            
            # Якщо кандидат хороший, пробуємо його виправити
            if current_score > max_score:
                max_score = current_score
                
                # --- ВИПРАВЛЕННЯ ---
                # Примусово міняємо типи даних згідно стандарту
                # 1. Літери (0,1,6,7)
                for idx in [0, 1, 6, 7]:
                    if segment[idx] in self.int_to_char: # Якщо там цифра, міняємо на літеру
                        segment[idx] = self.int_to_char[segment[idx]]
                
                # 2. Цифри (2,3,4,5)
                for idx in range(2, 6):
                    if segment[idx] in self.char_to_int: # Якщо там літера, міняємо на цифру
                        segment[idx] = self.char_to_int[segment[idx]]
                
                best_candidate = "".join(segment)

        # Повертаємо виправлений варіант, якщо score достатньо високий (хоча б 5 з 8 співпало)
        if max_score >= 5:
            return best_candidate
        return clean_text

    def enforce_eu_general(self, text):
        """Просто чистить сміття, не нав'язуючи жорстку структуру"""
        clean = self.clean(text)
        # Видаляємо типові префікси країн, якщо вони "прилипли"
        prefixes = ['UA', 'PL', 'LT', 'D', 'CZ', 'SK', 'II', '11']
        for p in prefixes:
            if clean.startswith(p) and len(clean) > len(p) + 4:
                return clean[len(p):]
        return clean

# Ініціалізація
enforcer = RegionEnforcer()

def fuzzy_check(detected_raw, database_str, country_mode="Auto"):
    """
    Головна функція перевірки з урахуванням країни
    """
    
    # 1. ОБРОБКА ЗАЛЕЖНО ВІД РЕГІОНУ
    processed_plate = detected_raw
    debug_info = f"Mode: {country_mode}"

    if country_mode == "Україна 🇺🇦":
        processed_plate = enforcer.enforce_ua(detected_raw)
        debug_info += " -> UA Enforced"
    elif country_mode == "Європа 🇪🇺":
        processed_plate = enforcer.enforce_eu_general(detected_raw)
        debug_info += " -> EU Cleaned"
    else:
        # Auto mode (стара логіка або базова чистка)
        processed_plate = enforcer.clean(detected_raw)
    
    # 2. ПЕРЕВІРКА В БАЗІ
    # Чистимо базу також
    db_list = [enforcer.clean(plate) for plate in database_str.split(',')]
    
    if processed_plate in db_list:
        return True, processed_plate, debug_info
    
    # Fuzzy match (на випадок 1 помилки)
    for db_plate in db_list:
        if len(db_plate) == len(processed_plate):
            diff = sum(1 for a, b in zip(db_plate, processed_plate) if a != b)
            if diff <= 1:
                return True, db_plate, f"{debug_info} (Fuzzy fix)"

    return False, processed_plate, debug_info