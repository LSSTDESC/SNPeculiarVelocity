# Dragan's nr library

## Installation

* in the src dir, run:
```
gcc -c *.c -I ../opt/local/include/recipes
```

* run the command
```
ar r librecipes.a *.o
```

* move `librecipes.a` into directory `/lib`

* in `/lib` run
```
ranlib librecipes.a
```
