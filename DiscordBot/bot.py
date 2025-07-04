# bot.py
import discord
from discord.ext import commands
import os
import json
import logging
import re
import requests
from report import Report, ReportType
from review import Review
import pdb
import sys

sys.path.append("../SuicidalIdeationDetector")
from train_classifier import BertClassifier

# Set up logging to the console
logger = logging.getLogger("discord")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename="discord.log", encoding="utf-8", mode="w")
handler.setFormatter(
    logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
)
logger.addHandler(handler)

# There should be a file called 'tokens.json' inside the same folder as this file
token_path = "tokens.json"
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens["discord"]


class ModBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=".", intents=intents)
        self.group_num = None
        self.mod_channels = {}  # Map from guild to the mod channel id for that guild
        self.reports = {}  # Map from user IDs to the state of their report
        self.reviews = {}  # Map from moderator IDs to the state of their review
        self.pending_reports = (
            {}
        )  # list of reports that haven't been reviewed yet. maps message to tuple of information regarding the report
        self.user_violations = {}  # Map from user IDs to number of violations recorded
        model_path = (
            "../SuicidalIdeationDetector/bert-base-uncased_suicide_classifier"
        )

        self.classifier = BertClassifier(model_name="bert-base-uncased")
        self.classifier.load_model(model_path)
        self.classifier_thresholds = {
            "low": 0.6,
            "medium": 0.8,
            "high": 0.9,
        }

    async def on_ready(self):
        print(f"{self.user.name} has connected to Discord! It is these guilds:")
        for guild in self.guilds:
            print(f" - {guild.name}")
        print("Press Ctrl-C to quit.")

        # Parse the group number out of the bot's name
        match = re.search("[gG]roup (\d+) [bB]ot", self.user.name)
        if match:
            self.group_num = match.group(1)
        else:
            raise Exception(
                'Group number not found in bot\'s name. Name format should be "Group # Bot".'
            )

        # Find the mod channel in each guild that this bot should report to
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name == f"group-{self.group_num}-mod":
                    self.mod_channels[guild.id] = channel

    async def on_message(self, message):
        """
        This function is called whenever a message is sent in a channel that the bot can see (including DMs).
        Currently the bot is configured to only handle messages that are sent over DMs or in your group's "group-#" channel.
        """
        # Ignore messages from the bot
        if message.author.id == self.user.id:
            return

        # Check if this message was sent in a server ("guild") or if it's a DM
        if message.guild:
            await self.handle_channel_message(message)
        else:
            await self.handle_dm(message)

    async def handle_dm(self, message):
        # Handle a help message
        if message.content == Report.HELP_KEYWORD:
            reply = "Use the `report` command to begin the reporting process.\n"
            reply += "Use the `cancel` command to cancel the report process.\n"
            await message.channel.send(reply)
            return

        if message.content.startswith("Classify: "):
            message_to_classify = message.content.split("Classify: ")[1]
            predicted_class, confidence = self.classifier.predict_text(message_to_classify)
            if predicted_class == 0:
                reply = (
                    f"Message: \"{message_to_classify}\" labeled as non-suicidal with confidence {confidence}.\n"
                )
            else:
                reply = (
                    f"Message: \"{message_to_classify}\" labeled as suicidal with confidence {confidence}.\n")

            await message.channel.send(reply)
            return

        author_id = message.author.id
        responses = []

        # Only respond to messages if they're part of a reporting flow
        if author_id not in self.reports and not message.content.startswith(
            Report.START_KEYWORD
        ):
            return

        # If we don't currently have an active report for this user, add one
        if author_id not in self.reports:
            self.reports[author_id] = Report(self)

        # Let the report class handle this message; forward all the messages it returns to uss
        responses = await self.reports[author_id].handle_message(message)
        for r in responses:
            await message.channel.send(r)

        # If the report is complete or cancelled, remove it from our map
        if self.reports[author_id].report_complete():
            completed_report = self.reports[author_id]
            if completed_report.needs_review:
                mod_channel = self.mod_channels[completed_report.message.guild.id]

                await mod_channel.send(
                    f'Requesting review for reported message:\n{completed_report.message.author.name}: "{completed_report.message.content}"\n'
                )
                await mod_channel.send(
                    f"Link to reported message:\n{completed_report.message.jump_url}"
                )
                if completed_report.is_emergency:
                    await mod_channel.send(
                        "\nThis report has indicated that this is an emergency and may require law enforcement involvement. Please review ASAP!"
                    )

            self.reports.pop(author_id)

    async def handle_channel_message(self, message):

        if message.channel.name == f"group-{self.group_num}-mod":
            await self.handle_mod_message(message)

        # Only handle messages sent in the "group-#" channel
        if message.channel.name == f"group-{self.group_num}":
            await self.handle_user_message(message)

        # Forward the message to the mod channel

        # mod_channel = self.mod_channels[message.guild.id]
        # await mod_channel.send(
        #     f'Forwarded message:\n{message.author.name}: "{message.content}"'
        # )
        # scores = self.eval_text(message.content)
        # await mod_channel.send(self.code_format(scores))

    async def handle_mod_message(self, message):
        if message.content == Review.HELP_KEYWORD:
            reply = "Use the `review` command to begin the mod review process.\n"
            reply += "Use the `cancel` command to cancel the review process.\n"
            await message.channel.send(reply)
            return

        author_id = message.author.id
        if author_id not in self.reviews and not message.content.startswith(
            Review.START_KEYWORD
        ):
            return

        # If we don't currently have an active review for this moderator, add one
        if author_id not in self.reviews:
            self.reviews[author_id] = Review(self)

        responses = await self.reviews[author_id].handle_message(message)
        for r in responses:
            await message.channel.send(r)

        if self.reviews[author_id].review_complete():
            self.reviews.pop(author_id)

    async def handle_user_message(self, message):
        print("handling user message!")
        message_text = message.content
        predicted_class, confidence = self.classifier.predict_text(message_text)
        if predicted_class == 0:
            print(
                f"Message: \"{message_text}\" labeled as non-suicidal with confidence {confidence}"
            )
        else:
            print(
                f"Message: \"{message_text}\" labeled as suicidal with confidence {confidence}"
            )
            if confidence > self.classifier_thresholds['low']:
                confidence_amount = "LOW" 
                if confidence > self.classifier_thresholds['medium']:
                    confidence_amount = "MEDIUM"
                if confidence > self.classifier_thresholds['high']:
                    confidence_amount = "HIGH"

                self.pending_reports[message] = {
                    "report_category": ReportType.SELF_HARM,
                    "report_sub_category": None,
                    "report_description": "The automatic flagging system has identified this message as potentially Self-Harm/Suicide related content",
                    "is_emergency": False,
                    "valid_emergency_count": 0,
                    "auto_flagged": True
                }
                mod_channel = self.mod_channels[message.guild.id]

                await mod_channel.send(
                    f'Requesting review for auto-flagged message:\n{message.author.name}: "{message.content}"\nClassifier Confidence: {confidence_amount}'
                )
                await mod_channel.send(
                    f"Link to reported message:\n{message.jump_url}"
                )

   

    def eval_text(self, message):
        """'
        TODO: Once you know how you want to evaluate messages in your channel,
        insert your code here! This will primarily be used in Milestone 3.
        """
        return message

    def code_format(self, text):
        """'
        TODO: Once you know how you want to show that a message has been
        evaluated, insert your code here for formatting the string to be
        shown in the mod channel.
        """
        return "Evaluated: '" + text + "'"


client = ModBot()
client.run(discord_token)
