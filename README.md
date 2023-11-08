# Muse

Extract music features and label your favorite songs for further model training.

## Usage

```
# open a nix shell with required dependencies
nix develop

# Launch a minimal test case with default parameters
./src/test.py
```

## Installation / Dependencies

### Scrapper Sources

## Contribution

### Project Structure

`src/ffmpeg_spectrogram.sh` is an unused relic.

## License

## TODO
* [ ] Clean up TODOs
* [x] Refactor MFExtractor class for tight CLI integration.
      (Future used as a helper script by some super structure.)
* [ ] Determine location of storage. XDG? Package-local? Global state?
* [ ] Reset training data. By age/genre/name/source/
      Plus renew/overwrite flag.
* [ ] Fully 'MFExtractor' class init by CLI passed args. Seed the members of @dataclass
* [ ] All this AI stuff man.
* [ ] Select only the most promising music features to train the model.
* [ ] Scramble input music or random noise as negative test against the models?
* [ ] VLC integration?
