import re
import os
import requests
from symspellpy import SymSpell, Verbosity
from num2words import num2words
from transformers import pipeline, logging

# CONFIGURATION
logging.set_verbosity_error()

class AdvancedCorrector:
    def __init__(self):
        # SYSTEM INITIALIZATION
        print("--- System Initialization ---")
        
        # MODEL LOADING
        print("[1/3] Loading NER Model (cahya/bert-base-indonesian-NER)...")
        try:
            self.ner_pipeline = pipeline(
                "token-classification", 
                model="cahya/bert-base-indonesian-NER", 
                tokenizer="cahya/bert-base-indonesian-NER",
                aggregation_strategy="simple" 
            )
        except Exception as e:
            print(f"    Failed to load BERT model: {e}")
            self.ner_pipeline = None

        # SYMSPELL SETUP
        print("[2/3] Setting up SymSpell Dictionary...")
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        # DATA STORAGE CONFIGURATION
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.dict_filename = os.path.join(self.data_dir, "full_dictionary_v7_suffix_stacking.txt")
        
        # DICTIONARY LOADING OR GENERATION
        if os.path.exists(self.dict_filename):
            print(f"    Loading cached database: {self.dict_filename}")
            self.sym_spell.load_dictionary(self.dict_filename, term_index=0, count_index=1)
        else:
            self._build_and_save_dictionary()

        self.changes_log = []

    # DATA SOURCES
    def _get_manual_cities(self):
        raw_cities = "Ambon, Balikpapan, Banda Aceh, Bandar Lampung, Bandung, Banjar, Banjarbaru, Banjarmasin, Batam, Batu, Baubau, Bekasi, Bengkulu, Bima, Binjai, Bitung, Blitar, Bogor, Bontang, Bukittinggi, Cilegon, Cimahi, Cirebon, Denpasar, Depok, Dumai, Gorontalo, Gunungsitoli, Jakarta Barat, Jakarta Pusat, Jakarta Selatan, Jakarta Timur, Jakarta Utara, Jambi, Jayapura, Kediri, Kendari, Kotamobagu, Kupang, Langsa, Lhokseumawe, Lubuk Linggau, Madiun, Magelang, Makassar, Malang, Manado, Mataram, Medan, Metro, Mojokerto, Padang, Padang Panjang, Padangsidempuan, Pagar Alam, Palangka Raya, Palembang, Palopo, Palu, Pangkalpinang, Parepare, Pariaman, Pasuruan, Payakumbuh, Pekalongan, Pekanbaru, Pematangsiantar, Pontianak, Prabumulih, Probolinggo, Sabang, Salatiga, Samarinda, Sawahlunto, Semarang, Serang, Sibolga, Singkawang, Solok, Sorong, Subulussalam, Sukabumi, Sungai Penuh, Surabaya, Surakarta, Tangerang, Tangerang Selatan, Tanjungbalai, Tanjungpinang, Tarakan, Tasikmalaya, Tebing Tinggi, Tegal, Ternate, Tidore Kepulauan, Tomohon, Tual, Yogyakarta"
        return [city.strip().lower() for city in raw_cities.split(',')]

    def _get_provinces_and_islands(self):
        raw_geo = "Aceh, Sumatera Utara, Sumatera Barat, Riau, Kepulauan Riau, Jambi, Bengkulu, Sumatera Selatan, Kepulauan Bangka Belitung, Lampung, Banten, DKI Jakarta, Jawa Barat, Jawa Tengah, DI Yogyakarta, Jawa Timur, Bali, Nusa Tenggara Barat, Nusa Tenggara Timur, Kalimantan Barat, Kalimantan Tengah, Kalimantan Selatan, Kalimantan Timur, Kalimantan Utara, Sulawesi Utara, Gorontalo, Sulawesi Tengah, Sulawesi Barat, Sulawesi Selatan, Sulawesi Tenggara, Maluku, Maluku Utara, Papua, Papua Barat, Papua Selatan, Papua Tengah, Papua Pegunungan, Papua Barat Daya, Sumatera, Jawa, Kalimantan, Sulawesi, Papua, Bali, Lombok, Sumbawa, Flores, Sumba, Timor, Halmahera, Seram, Buru, Bangka, Belitung, Nias, Mentawai, Madura"
        return [geo.strip().lower() for geo in raw_geo.split(',')]

    def _get_common_particles(self):
        return ["saya", "aku", "ku", "hamba", "kami", "kita", "kamu", "engkau", "kau", "anda", "kalian", "saudara", "dia", "ia", "beliau", "mereka", "nya", "ini", "itu", "sini", "situ", "sana", "apa", "siapa", "mana", "kapan", "mengapa", "kenapa", "bagaimana", "berapa", "di", "ke", "dari", "pada", "dalam", "atas", "bawah", "kepada", "daripada", "untuk", "bagi", "guna", "buat", "oleh", "dengan", "tentang", "mengenai", "terhadap", "soal", "sejak", "semenjak", "sampai", "hingga", "keluar", "masuk", "dan", "serta", "atau", "tetapi", "tapi", "namun", "melainkan", "sedangkan", "jika", "kalau", "jikalau", "asal", "bila", "manakala", "agar", "supaya", "biar", "sebab", "karena", "lantaran", "sehingga", "maka", "akibatnya", "ketika", "sewaktu", "tatkala", "selagi", "seraya", "sambil", "setelah", "sesudah", "sebelum", "sehabis", "selesai", "bahwa", "yakni", "yaitu", "adalah", "ialah", "merupakan", "biarpun", "meskipun", "walaupun", "sekalipun", "sungguhpun", "padahal", "kendatipun", "kah", "lah", "tah", "pun", "per", "yang", "tak", "tidak", "bukan", "tanpa", "tiada", "belum", "sudah", "telah", "akan", "sedang", "lagi", "pernah", "masih", "baru", "ada", "bisa", "dapat", "boleh", "harus", "mesti", "wajib", "perlu", "butuh", "mau", "ingin", "hendak", "bakal", "sangat", "amat", "terlalu", "paling", "cukup", "kurang", "lebih", "agak", "hanya", "cuma", "saja", "juga", "pun", "nanti", "kemarin", "besok", "lusa", "sekarang", "dahulu", "dulu", "tadi", "barusan", "tentu", "pasti", "yakin", "memang", "barangkali", "mungkin", "bahkan", "malah", "justru", "segera", "langsung", "lantas", "kemudian", "lalu", "akhirnya", "pak", "bapak", "bu", "ibu", "mas", "mbak", "kak", "kakak", "bang", "abang", "dik", "adik", "om", "tante"]

    # MORPHOLOGICAL GENERATION LOGIC
    def _apply_morphology(self, root):
        forms = set()
        forms.add(root)

        first = root[0]
        second = root[1] if len(root) > 1 else ""
        is_vowel = second in 'aiueo'

        def get_nasal_root(prefix, r):
            if prefix in ['me', 'pe']:
                if first == 'k': return prefix + 'ng' + (r[1:] if is_vowel else r)
                elif first == 'p': return prefix + 'm' + (r[1:] if is_vowel else r)
                elif first == 's': return prefix + 'ny' + (r[1:] if is_vowel else r)
                elif first == 't': return prefix + 'n' + (r[1:] if is_vowel else r)
                elif first in 'lmnrwy': return prefix + r
                elif first in 'cdjz': return prefix + 'n' + r
                elif first in 'gh' or (first=='k' and second=='h'): return prefix + 'ng' + r
                elif first in 'aiueo': return prefix + 'ng' + r
            return prefix + r

        def get_ber_ter_root(prefix, r):
            if r.startswith('r'): return prefix[:-1] + r
            return prefix + r

        # LEVEL 1: BASIC PREFIXES
        level1_bases = set()
        level1_bases.add(root)
        
        base_me = get_nasal_root('me', root)
        base_pe = get_nasal_root('pe', root)
        level1_bases.add(base_me)
        level1_bases.add(base_pe)
        
        level1_bases.add('di' + root)
        level1_bases.add('ke' + root)
        level1_bases.add('se' + root)
        
        base_ber = get_ber_ter_root('ber', root)
        base_ter = get_ber_ter_root('ter', root)
        base_per = get_ber_ter_root('per', root)
        level1_bases.add(base_ber)
        level1_bases.add(base_ter)
        level1_bases.add(base_per)

        forms.update(level1_bases)

        # LEVEL 2: TRANSITIVE/NOUN SUFFIXES
        level2_bases = set()
        suffixes_transitive = ['kan', 'i']
        suffix_noun = ['an']

        for base in level1_bases:
            noun_form = base + 'an'
            level2_bases.add(noun_form)
            
            if base.startswith(('me', 'di', 'ter')):
                for suf in suffixes_transitive:
                    level2_bases.add(base + suf)

        forms.update(level2_bases)

        # LEVEL 3: ENCLITICS (SUFFIX STACKING)
        enclitics = ['nya', 'ku', 'mu']
        particles = ['lah', 'kah', 'pun']

        candidates_for_enclitics = level1_bases.union(level2_bases)

        for base in candidates_for_enclitics:
            for enc in enclitics:
                forms.add(base + enc)
            
            for part in particles:
                forms.add(base + part)

        return forms

    # DICTIONARY CONSTRUCTION
    def _build_and_save_dictionary(self):
        print("    [INFO] Building Ultimate V7 Database (Suffix Stacking)...")
        final_dictionary = set()

        url_dataset = "https://raw.githubusercontent.com/sastrawi/sastrawi/master/data/kata-dasar.txt"
        print("    Downloading root words...")
        try:
            r = requests.get(url_dataset)
            kata_dasar_list = [line.strip().lower() for line in r.text.splitlines() if line.strip()]
        except Exception as e:
            print(f"    [ERROR] {e}")
            kata_dasar_list = []

        print(f"    Generating variations for {len(kata_dasar_list)} root words...")
        for idx, root in enumerate(kata_dasar_list):
            variations = self._apply_morphology(root)
            final_dictionary.update(variations)
            if idx % 5000 == 0: print(f"    ... {idx} words processed")

        extras = self._get_manual_cities() + self._get_provinces_and_islands() + self._get_common_particles()
        for item in extras:
            for part in item.split(): final_dictionary.add(part)

        custom_entities = ["rupiah", "hobi", "proyek", "triliun", "miliar", "juta", "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]
        final_dictionary.update(custom_entities)

        if os.path.exists(self.dict_filename): os.remove(self.dict_filename)
        print(f"    Saving {len(final_dictionary)} words to {self.dict_filename}...")
        
        with open(self.dict_filename, "w", encoding="utf-8") as f:
            for word in final_dictionary:
                if re.match(r'^[a-z\-]+$', word):
                    f.write(f"{word} 1\n")
        
        self.sym_spell.load_dictionary(self.dict_filename, term_index=0, count_index=1)
        print("    Database construction complete.")

    def log_change(self, type_err, original, fixed):
        if original != fixed:
            self.changes_log.append({"Type": type_err, "Original": original, "Fixed": fixed})

    # CORRECTION PIPELINE
    def fix_punctuation_spacing(self, text):
        text = re.sub(r'\s+([.,;:?!])', r'\1', text)
        return re.sub(r'([.,;:?!])(?=[a-zA-Z])', r'\1 ', text)

    def fix_reduplication(self, text):
        pattern = r'\b([a-zA-Z]+) \1\b'
        def replacement(match):
            fixed = f"{match.group(1)}-{match.group(1)}"
            self.log_change("Reduplication", match.group(0), fixed)
            return fixed
        return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    def fix_kpst_correction_pre(self, text):
        words = text.split()
        fixed_words = []
        for word in words:
            original = word
            clean_word = word.lower()
            if re.match(r'^memp[aiueo]', clean_word):
                fixed = re.sub(r'^memp', 'mem', clean_word); self.log_change("KPST Correction", original, fixed); fixed_words.append(fixed); continue
            if re.match(r'^ment[aiueo]', clean_word):
                fixed = re.sub(r'^ment', 'men', clean_word); self.log_change("KPST Correction", original, fixed); fixed_words.append(fixed); continue
            if re.match(r'^mens[aiueo]', clean_word):
                fixed = re.sub(r'^mens', 'meny', clean_word); self.log_change("KPST Correction", original, fixed); fixed_words.append(fixed); continue
            if re.match(r'^mengk[aiueo]', clean_word):
                fixed = re.sub(r'^mengk', 'meng', clean_word); self.log_change("KPST Correction", original, fixed); fixed_words.append(fixed); continue
            if 'menpegang' in clean_word:
                 fixed = clean_word.replace('menpegang', 'memegang'); self.log_change("KPST/Typo Correction", original, fixed); fixed_words.append(fixed); continue
            fixed_words.append(word)
        return " ".join(fixed_words)

    def fix_spelling_advanced(self, text):
        protected_ranges = []
        if self.ner_pipeline:
            ner_results = self.ner_pipeline(text)
            for entity in ner_results:
                if entity['entity_group'] in ['PER', 'ORG'] and entity['score'] > 0.5:
                    protected_ranges.append(range(entity['start'], entity['end']))

        tokens = []
        for match in re.finditer(r'\S+', text): 
            tokens.append((match.group(), match.start(), match.end()))
        
        fixed_text_parts = []
        last_end = 0

        for word, start, end in tokens:
            fixed_text_parts.append(text[last_end:start])
            
            is_protected = False
            token_range = range(start, end)
            for r in protected_ranges:
                if set(token_range).intersection(set(r)):
                    is_protected = True; break
            
            if is_protected:
                fixed_text_parts.append(word); last_end = end; continue

            clean_word = re.sub(r'[^\w\-]', '', word) 
            
            if not clean_word or any(c.isdigit() for c in clean_word):
                fixed_text_parts.append(word); last_end = end; continue
            
            if self.sym_spell.lookup(clean_word.lower(), Verbosity.TOP, max_edit_distance=0):
                fixed_text_parts.append(word); last_end = end; continue
            
            if '-' in clean_word:
                parts = clean_word.split('-')
                is_valid_reduplication = True
                for part in parts:
                    if not part: continue
                    if not self.sym_spell.lookup(part.lower(), Verbosity.TOP, max_edit_distance=0):
                        is_valid_reduplication = False; break
                if is_valid_reduplication:
                    fixed_text_parts.append(word); last_end = end; continue

            suggestions = self.sym_spell.lookup(clean_word.lower(), Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                best = suggestions[0].term
                if len(clean_word) < 4 and len(best) != len(clean_word): 
                    fixed_text_parts.append(word)
                else:
                    fixed = best.title() if word[0].isupper() else best
                    if not word[-1].isalnum(): fixed += word[-1]
                    self.log_change("Spelling", word, fixed)
                    fixed_text_parts.append(fixed)
            else: 
                fixed_text_parts.append(word)
            
            last_end = end
            
        fixed_text_parts.append(text[last_end:])
        return "".join(fixed_text_parts)

    def fix_numbers_eyd(self, text):
        def is_part_of_list(full_text, start, end):
            window = full_text[max(0, start-20):min(len(full_text), end+20)]
            return ',' in window and ('dan' in window or re.search(r'\d', window))
        def num_replacer(match):
            num_str = match.group(0)
            clean = num_str.replace('.', '').replace(',', '')
            if not clean.isdigit(): return num_str
            num = int(clean)
            try:
                words = num2words(num, lang='id')
                if len(words.split()) == 1 and not is_part_of_list(text, match.start(), match.end()):
                    self.log_change("Num to Word", num_str, words); return words
            except: pass
            suffixes = {1000000000000: 'triliun', 1000000000: 'miliar', 1000000: 'juta', 1000: 'ribu'}
            for l, label in suffixes.items():
                if num >= l and num % l == 0:
                    res = f"{num // l} {label}"; self.log_change("Large Num", num_str, res); return res
            return num_str
        return re.sub(r'\b\d[\d.]*\b', num_replacer, text)

    def fix_capitalization_ner(self, text):
        if not self.ner_pipeline: return text
        results = self.ner_pipeline(text)
        text_chars = list(text)
        results.sort(key=lambda x: x['start'], reverse=True)
        allowed_tags = ['PER', 'ORG', 'LOC', 'GPE'] 
        for entity in results:
            label = entity['entity_group']
            score = entity['score']
            if label in allowed_tags and score > 0.6: 
                start, end = entity['start'], entity['end']
                word = text[start:end]
                if word[0].isupper(): continue
                fixed_word = word.title()
                for i in range(len(fixed_word)): text_chars[start + i] = fixed_word[i]
                self.log_change(f"Capitalization ({label})", word, fixed_word)
        text = "".join(text_chars)
        return re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

    def process(self, text):
        self.changes_log = []
        print("\n[INFO] Processing Text...")
        text = self.fix_punctuation_spacing(text)
        text = self.fix_reduplication(text)
        text = self.fix_kpst_correction_pre(text)
        text = self.fix_spelling_advanced(text) 
        text = self.fix_numbers_eyd(text)
        text = self.fix_capitalization_ner(text)
        return text