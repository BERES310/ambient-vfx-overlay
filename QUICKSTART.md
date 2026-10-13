# Quick Start

Turn a photo into a looping weather video (rain, snow, mist, sunbeams, storm).

## 1. Start the app
Double-click **Start.bat**. The first time it creates a private Python environment and installs
what it needs (several minutes). Everything stays inside this folder.

## 2. Make a video (three steps)
1. **Pick a photo** – click a tile, or *+ Add photo*. Put your own photos in `input_images/`.
2. **Pick a mood** – Rainy, Snowy, Sunny, Stormy or Misty dawn. A small live preview plays
   within a few seconds. Use the *How much / Wind / Darkness* sliders to taste.
3. **Press "Make final video"** – choose *Quality* and *Length* first. The button shows a time
   estimate that gets accurate after your first video.

Tip: start with quality **Draft** and length **10 seconds** to check the look, then go up.

Finished videos are in `output/live_wallpapers/` and on the **My videos** tab, where
*Use this look* reloads the exact settings that made a video.

## Stopping and continuing
Press **Stop** (or Esc) any time. Next time the app offers **Continue where you stopped** –
nothing is lost, even after closing the window or a power cut.

## Using it as a wallpaper
Windows cannot play a video as the desktop by itself. After a render, press
**Wallpaper Engine folder** and copy the folder it opens into Wallpaper Engine's
`projects/myprojects/` folder (Lively also plays the mp4 directly). *Set as wallpaper* sets the
first picture only.

## Command line (batches, overnight)
```
python vfx_cli.py photo.jpg --mood storm --quality draft --length 10
python vfx_cli.py a.jpg b.png --mood rain snow      # 4 videos, one summary at the end
python vfx_cli.py --all-photos --mood mist
python vfx_cli.py --help
```
Defaults live in `settings.json` (mood, quality, length, fps, intensity, wind, darkness).
A wrong value is reported with the allowed range.

## Troubleshooting
| Symptom | Fix |
|---|---|
| "The AI models are missing" | *Settings* tab → **Download missing AI models** (about 20 GB, needs internet once). |
| "Graphics card ran out of space" | Choose **Draft** or **Balanced** quality, or a smaller photo, or fewer effects. |
| No preview / very slow | No compatible GPU was found; the app then defaults to Draft. Update your graphics driver. |
| "Encoder (ffmpeg) not found" | Run Start.bat again to reinstall the requirements. |
| "File in use" when saving | Close any player showing the previous video. |
| The window seems frozen | The final video is being made – watch the progress bar. Previews pause during it. |
| Anything else | *Copy details* in the error box, or open `output/gui_error.log`. |

## What each effect does
- **Rain** – falling streaks that pass behind near objects, with splashes and wet surfaces.
- **Snow** – drifting flakes with depth blur.
- **Mist** – low fog that hugs the ground and fades with distance.
- **Lightning** – occasional flashes that light the whole scene.
- **Light flicker** – lamps and windows pulse slightly.
- **Sunbeams** – soft rays from the brightest area of the sky.
