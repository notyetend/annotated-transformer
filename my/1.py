def utf16_decimals(char, chunk_size=2):
    # encode the character as big-endian utf-16
    encoded_char = char.encode('utf-16-be')

    # convert every `chunk_size` bytes to an integer
    decimals = []
    for i in range(0, len(encoded_char), chunk_size):
        chunk = encoded_char[i:i+chunk_size]
        decimals.append(int.from_bytes(chunk, 'big'))

    return decimals


class JamosSeparator:
    def __init__(self, string):
        self.string = string
        self.result = []
        self.choseong_list = [char for char in "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"]
        self.jungseong_list = [char for char in "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"]
        self.jongseong_list = [char for char in " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]

    def run(self):
        for char in self.string:

            character_code = ord(char)

            # Do not process unless it is in Hangul syllables range.
            if 0xD7A3 < character_code or character_code < 0xAC00:
                continue

            choseong_index = (character_code - 0xAC00) // 21 // 28
            jungseong_index = (character_code - 0xAC00 - (choseong_index * 21 * 28)) // 28
            jongseong_index = character_code - 0xAC00 - (choseong_index * 21 * 28) - (jungseong_index * 28)

            self.result.append(self.choseong_list[choseong_index])
            self.result.append(self.jungseong_list[jungseong_index])
            self.result.append(self.jongseong_list[jongseong_index])
            self.result.append("_")

    def get(self):
        return self.result


js = JamosSeparator('밥')
js.run()
js.get()
for i in '밥':
    print(ord(i))