import telegram
import telegram.ext
from PIL import Image
from model import Model
from secret import *

USAGE_TEXT = (
    "Send me some photos. "
    "I'll detect objects on them. "
    "You can use your camera or "
    "inline bots like @bing and @pic.")

def handle_text(update, context):
    update.message.reply_text(USAGE_TEXT)

def handle_photo(update, context):
    
    user = update.message.from_user
    chat_id = update.message.chat_id
    file_id = update.message.photo[-1]['file_id']
    context.bot.getFile(file_id).download('in.jpg')
    model.detect(Image.open('in.jpg')).save('out.jpg')

    with open('out.jpg','rb') as fd:
        context.bot.send_photo(chat_id, fd)
        if user['id'] != TG_BOT_OWNER_ID:
            msg = f"@{user['username']} {user['id']}"
            context.bot.send_photo(TG_BOT_OWNER_ID, fd, msg)

ft = telegram.ext.Filters.text
fp = telegram.ext.Filters.photo
h = telegram.ext.MessageHandler
u = telegram.ext.Updater(TG_BOT_TOKEN)
model = Model('yolov5s.onnx', 'labels.txt')
u.dispatcher.add_handler(h(ft, handle_text))
u.dispatcher.add_handler(h(fp, handle_photo))
u.start_polling(); u.idle()