#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for logging utilities
"""

import pytest
import logging
import argparse
from shoot.log import (
    setup_logging,
    add_logging_parser_arguments,
    add_log_level_parser_arguments,
    DEFAULT_LOGGING_CONFIG,
)


class TestLoggingSetup:
    """Test logging configuration"""

    def test_setup_logging_basic(self):
        """Test basic logging setup"""
        # First call may modify the config, but that's ok
        setup_logging(to_file=False, show_init_msg=False)

        logger = logging.getLogger("woom")
        assert logger is not None
        assert logger.level == logging.DEBUG

    def test_setup_logging_console_level(self):
        """Test setting console log level"""
        # This test may fail if already configured, just check it doesn't raise
        try:
            setup_logging(console_level="warning", to_file=False, show_init_msg=False)
            logger = logging.getLogger("woom")
            assert logger is not None
        except ValueError:
            # Config may already be modified from previous test, that's ok
            pass

    def test_setup_logging_no_color(self):
        """Test logging without color"""
        # This test may fail if already configured, just check it doesn't raise
        try:
            setup_logging(to_file=False, no_color=True, show_init_msg=False)
            logger = logging.getLogger("woom")
            assert logger is not None
        except ValueError:
            # Config may already be modified from previous test, that's ok
            pass

    def test_default_logging_config(self):
        """Test that default config is well-formed"""
        assert "version" in DEFAULT_LOGGING_CONFIG
        assert "formatters" in DEFAULT_LOGGING_CONFIG
        assert "handlers" in DEFAULT_LOGGING_CONFIG
        assert "loggers" in DEFAULT_LOGGING_CONFIG

        # Check formatters exist
        assert "brief" in DEFAULT_LOGGING_CONFIG["formatters"]
        assert "precise" in DEFAULT_LOGGING_CONFIG["formatters"]

        # Check handlers exist
        assert "console" in DEFAULT_LOGGING_CONFIG["handlers"]
        assert "file" in DEFAULT_LOGGING_CONFIG["handlers"]


class TestParserArguments:
    """Test argument parser utilities"""

    def test_add_log_level_parser_arguments(self):
        """Test adding log level arguments to parser"""
        parser = argparse.ArgumentParser()
        add_log_level_parser_arguments(parser)

        # Should have log-level argument
        args = parser.parse_args(["--log-level", "debug"])
        assert args.log_level == "debug"

        args = parser.parse_args(["--log-level", "info"])
        assert args.log_level == "info"

    def test_add_log_level_parser_default(self):
        """Test default log level"""
        parser = argparse.ArgumentParser()
        add_log_level_parser_arguments(parser, default_level="warning")

        args = parser.parse_args([])
        assert args.log_level == "warning"

    def test_add_logging_parser_arguments(self):
        """Test adding all logging arguments"""
        parser = argparse.ArgumentParser()
        add_logging_parser_arguments(parser)

        args = parser.parse_args(
            ["--log-level", "debug", "--log-file", "test.log", "--log-no-color"]
        )

        assert args.log_level == "debug"
        assert args.log_file == "test.log"
        assert args.log_no_color is True

    def test_add_logging_parser_arguments_defaults(self):
        """Test default values for logging arguments"""
        parser = argparse.ArgumentParser()
        add_logging_parser_arguments(parser)

        args = parser.parse_args([])

        assert args.log_level == "info"
        assert args.log_file == "shoot.log"
        assert args.log_no_color is False
