<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

Here is a custom version of detectron2 (based on 0.2.1).

### Updates
* Forked from [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) (19/08/2020)
* Add [FCOS](https://arxiv.org/abs/1904.01355) project, add `cfg.SOLVER.LR_MIN` for `WarmupCosineLR` (20/08/2020)


## Installation (official)

See [INSTALL.md](INSTALL.md).

## Installation (win10)

See [INSTALL_win10.md](INSTALL_win10.md).

## Highlights

1. Support [FCOS](https://arxiv.org/abs/1904.01355). [For more](projects/FCOS/fcos)
2. Support [Centernet](https://arxiv.org/abs/1904.07850) under building... [For more](projects/FCOS/centernet) 

## Future
1. Add support of tfrecord for dataloader.
2. Add multiple data augmentation, e.g. mosaic, label smooth....

## Changed files list

Files chagned relative to official code. See [CHANGED_FILES.md](CHANGED_FILES.md)

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
