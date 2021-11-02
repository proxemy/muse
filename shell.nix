#!/usr/bin/env nix-shell

{ pkgs ? import <nixpkgs> {} }:
	pkgs.mkShell {
		nativeBuildInputs = with pkgs.buildPackages.python3Packages; [
			librosa
			numpy
			matplotlib
		];
	}
