# Conv_OPTEE

This repository contains the code to perfom depthwise partitioning of the final layers of a neural network to deploy parts of it on the secure world.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

TAforNN is a demonstration of how to leverage OP-TEE, an open-source Trusted Execution Environment, to perform inference operations of a neural network. This example provides a starting point for developers interested in utilizing OP-TEE for similar tasks.

## Installation

To use TAforNN, follow these steps:

1. Install OPTEE as the secure OS and linux as the non-secure OS on your target machine
2. Clone this repository and copy it inside /optee_example


## Usage

To run TA for NN, execute the following steps:


1. Load the OP-TEE environment.
2. Execute optee_convolution in the normal world





