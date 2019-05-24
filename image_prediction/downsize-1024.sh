mkdir -p 1024/image_dir_processed
find image_dir_processed/ -iname "*.png" | parallel convert -resize "1024x1024^" {} 1024/{}
