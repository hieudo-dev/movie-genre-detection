import eel
from src import hooks
eel.init('web', allowed_extensions=['.js'])
eel.start('index.html')

