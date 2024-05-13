class FileUtils:
    _parent_address = "assets/"

    @staticmethod
    def write_to_txt(name: str, text: str):
        try:
            with open(FileUtils._parent_address + name, 'w') as file:
                file.write(text)
        except IOError:
            print("Dosya yazma hatası!")
