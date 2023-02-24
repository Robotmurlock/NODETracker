ASSETS_PATH="../../assets"

mkdir -p $ASSETS_PATH
wget -P $ASSETS_PATH --no-check-certificate https://motchallenge.net/data/MOT20Labels.zip # No images currently
unzip $ASSETS_PATH/MOT20Labels.zip -d $ASSETS_PATH
mv $ASSETS_PATH/MOT20Labels $ASSETS_PATH/MOT20
mkdir $ASSETS_PATH/MOT20/val
mv $ASSETS_PATH/MOT20/train/MOT20-05 $ASSETS_PATH/MOT20/val/MOT20-05
