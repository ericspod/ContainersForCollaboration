#! python

# (c) 2020 Eric Kerfoot, see LICENSE file

# This script is for applying inference to a nifti file and saving the output to a new nifti file. This can be used
# as a standalone script or in a Docker container. It relies on the network to use for inference to be saved to net.zip
# by default, change this by setting NET_NAME to some other path. 

import sys
import argparse
import gzip

import torch
import numpy as np
import nibabel as nib

NET_NAME='./net.zip'


def rescale_array(arr, dtype=np.float32, eps=1e-10):
    """Rescale the values of numpy array `arr' to the range unit range."""
    if dtype is not None:
        arr = arr.astype(dtype)

    mina = np.min(arr)
    maxa = np.max(arr)

    return (arr - mina) / max(eps, maxa - mina)


def infer(net, device, inputs):
    """Apply `net` to inputs, which is assumed to be a 2D image."""

    inputs = rescale_array(inputs)  # rescale the values of inputs to [0,1]

    # convert to a 4D tensor, ie. a batch of 1 with a single channel
    inputst = torch.from_numpy(inputs[None, None]).to(device)

    _, preds = net(inputst)  # network returns logits and predicted segmentations

    preds = preds[0].cpu().data.numpy()  # convert to a 2D numpy array

    return preds


def infer_volume(net, device, vol):
    """
    Apply inference with `net` to the volume `vol`, which is either 2D (WH), 3D (WHD), or 4D (WHDT).
    """
    output = np.zeros(vol.shape, np.int32)

    if vol.ndim == 2:
        output[...] = infer(net, device, vol)
    elif vol.ndim == 3:
        for d in range(vol.shape[2]):
            output[..., d] = infer(net, device, vol[..., d])
    elif vol.ndim == 4:
        for t in range(vol.shape[3]):
            for d in range(vol.shape[2]):
                output[..., d, t] = infer(net, device, vol[..., d, t])
    else:
        raise ValueError(f"Input value `vol` must be 2/3/4D, got shape {vol.shape}")

    return output


def load_nifti(filename):
    """Load the Nifti file from the given filename, or from stdin if `filename` is None."""

    # read the nifti file as data from stdin or the given file
    if filename is not None:
        return nib.load(filename)
    else:
        imdata = sys.stdin.buffer.read()

        # attempt to decompress the nifti file, a raised OSError means the file wasn't compressed
        try:
            imdata = gzip.decompress(imdata)  
        except OSError:
            pass

        return nib.nifti1.Nifti1Image.from_bytes(imdata)


def save_nifti(filename, output, affine, header):
    """Save the image volume `output` to a nifti file `filename` if this isn't None, to stdout otherwise."""

    imout = nib.nifti1.Nifti1Image(output, affine, header)

    if filename is not None:
        nib.save(imout, filename)
    else:
        sys.stdout.buffer.write(imout.to_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="Optional input file, read from stdin if not given", nargs="?")
    parser.add_argument("out_file", help="Optional output file, write to stdout if not given", nargs="?")
    args = parser.parse_args()

    # load the network, assigning it to the selected device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = torch.jit.load(NET_NAME, map_location=device)

    imloaded = load_nifti(args.in_file)
    
    output = infer_volume(net, device, imloaded.get_fdata())

    save_nifti(args.out_file, output, imloaded.affine, imloaded.header)
