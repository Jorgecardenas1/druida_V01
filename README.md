# DRUIDA
## _The master intelligence for metasurface design_

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

Druida is an artificial intelligence developed support the metasurfaces design process.

- Generative pipelines for metasurfaces design
The goal is to provide a stable version of the most important algorithmic pipelines to train and deploy AI for metasurfaces design.

## Features

- Deep Neural Network Stack
- GAN Generator Stack
- GAN Discriminator Stack
- Unconditional Diffusion Model
- Conditional Diffusion Model

## Goals
> Configurable AI models
> Easy to interface and use through jupyter notebooks.
> Reproduceable models
> API to future hyperparameters optimization



## Tech

Dillinger uses a number of open source projects to work properly:

- [Python] - Python 3.
- [PyTorch] - The framework to build our algorithms.
- [CLIP] - Pipelione to produce word encoding.


## Installation


Install the dependencies and devDependencies and start the server.
https://pypi.org/project/druida/

```sh
pip install druida

```

## Plugins

Dillinger is currently extended with the following plugins.
Instructions on how to use them in your own application are linked below.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantaneously see your updates!

Open your favorite Terminal and run these commands.

First Tab:

```sh
node app
```

Second Tab:

```sh
gulp watch
```

(optional) Third:

```sh
karma test
```

#### Building for source

For production release:

```sh
gulp build --prod
```

Generating pre-built zip archives for distribution:

```sh
gulp build dist --prod
```


MIT

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
