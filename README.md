# batch-font-runner

## Collecting a list of fonts

For example using the google fonts repo:

```sh
find $HOME/src/fonts/{ofl,apache,ufl}/ -iname "*.[ot]tf" > fonts.txt
```

## run_hb_depend_parity_check.py

Helper to run `hb-depend-closure-parity` on a collection of fonts in parallel. Reports any found under
approximations to stdout. Example execution:

```sh
python3 run_hb_depend_parity_check.py  -n 100 -j 90 --checker-path=$HOME/src/harfbuzz/build/test/fuzzing/hb-depend-closure-parity fonts.txt > under-approx.txt
```

Sample output:

```
UNDER-APPROX on ../fonts/ofl/padauk/Padauk-Bold.ttf for test ID 0xeddbddde0ee2ed74
UNDER-APPROX on ../fonts/ofl/padauk/Padauk-Regular.ttf for test ID 0xeddbddde0ee2ed74
...
```
