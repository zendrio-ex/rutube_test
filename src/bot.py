import telebot
import model
from utils import preparation
from torchvision import transforms
from config import settings
from io import BytesIO
from PIL import Image

model_obj = model.install_model()
labels = model.labels

bot = telebot.TeleBot(settings['bot_token'])


@bot.message_handler(content_types=['text'])
def text_message(message):
    bot.send_message(message.chat.id, "Please, send a photo of food.")


@bot.message_handler(content_types='photo')
def photo_message(message):
    fileID = message.photo[-1].file_id
    downloaded_file = bot.download_file(bot.get_file(fileID).file_path)

    image = Image.open(BytesIO(downloaded_file))
    image = transforms.ToTensor()(image)
    
    image = preparation.pipeline(image)

    label = model_obj(image.unsqueeze(dim=0))
    print(label)
    
    label = label.argmax(dim=-1).item()
    print(label)
    label = labels[label]

    bot.send_message(message.chat.id,
                     f"I think ... this is {' '.join(label.split('_'))}")


if __name__ == '__main__':
    bot.polling()
