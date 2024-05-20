class FileUtils:
    _parent_address = "assets/models/"

    @staticmethod
    def write_to_txt(name: str, text: str):
        try:
            with open(FileUtils._parent_address + name, 'w') as file:
                file.write(text)
        except IOError:
            print("Dosya yazma hatası!")

    @staticmethod
    def read_from_txt(name: str):
        try:
            with open(FileUtils._parent_address + name, 'r') as file:
                data = file.read()
            return data
        except IOError:
            print("Dosya okuma hatası!")