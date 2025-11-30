#!/bin/bash

echo "Converting WAV to MP3 using FFmpeg..."

for f in *.wav; do
    ffmpeg -y -i "$f" -codec:a libmp3lame -b:a 192k "${f%.wav}.mp3"
done

echo "Done."