from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

#APIキーの情報

key = "8f5ce89cd1ae908fc1e1b0b21d19ae9e"
secret = "6aa1b7db14e354ab"
wait_time = 1

#保存フォルダの指定
animalname = sys.argv[1]
savedir = "./"+ animalname

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = animalname,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    sefe_search = 1,
    extras = 'url_q, licence'
)

photos = result['photos']
#返り値を表示する
#pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
