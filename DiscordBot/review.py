from enum import Enum, auto
import discord
import re

class Outcome(Enum):
    TEMP_BAN = auto()
    PERMANENT_BAN = auto()
    DELETE_MESSAGE = auto()

class State(Enum):
    REVIEW_START = auto()
    AWAITING_MESSAGE = auto()
    MESSAGE_IDENTIFIED = auto()
    REVIEW_COMPLETE = auto()

class Review:
    START_KEYWORD = "review"
    CANCEL_KEYWORD = "cancel"
    HELP_KEYWORD = "help"

    def __init__(self, client):
        self.state = State.REVIEW_START
        self.client = client
        self.message = None

    async def handle_message(self, message):

        if message.content == self.CANCEL_KEYWORD:
            self.state = State.REVIEW_COMPLETE
            return ["Review cancelled."]
        
        if self.state == State.REVIEW_START:
            reply =  "Thank you for starting the review process. "
            reply += "Say `help` at any time for more information.\n\n"
            reply += "Please copy paste the link to the message you want to review.\n"
            reply += "You can obtain this link by right-clicking the message and clicking `Copy Message Link`."
            self.state = State.AWAITING_MESSAGE

            return [reply]
        
        if self.state == State.AWAITING_MESSAGE:
            # Parse out the three ID strings from the message link
            m = re.search('/(\d+)/(\d+)/(\d+)', message.content)
            if not m:
                return ["I'm sorry, I couldn't read that link. Please try again or say `cancel` to cancel."]
            guild = self.client.get_guild(int(m.group(1)))
            if not guild:
                return ["I cannot accept reviews of messages from guilds that I'm not in. Please have the guild owner add me to the guild and try again."]
            channel = guild.get_channel(int(m.group(2)))
            if not channel:
                return ["It seems this channel was deleted or never existed. Please try again or say `cancel` to cancel."]
            try:
                message = await channel.fetch_message(int(m.group(3)))
            except discord.errors.NotFound:
                return ["It seems this message was deleted or never existed. Please try again or say `cancel` to cancel."]

            if message not in self.client.pending_reports:
              return ["It seems this message was never reported or has already been reviewed. Please try again or say `cancel` to cancel."]

            self.message = message

            self.state = State.MESSAGE_IDENTIFIED
            return ["How would you like to handle this message?"]
            
        
        if self.state == State.MESSAGE_IDENTIFIED:
            return ["<insert rest of review flow here>"]

        return []

    def end_report(self):
        self.client.pending_reports.remove(self.message)
        self.state == State.REVIEW_COMPLETE
        self.message = None

    def review_complete(self):
        return self.state == State.REVIEW_COMPLETE
    
    async def handle_outcome(self, outcome_type):
        if outcome_type == Outcome.DELETE_MESSAGE:
            await self.message.delete()
        elif outcome_type == Outcome.TEMP_BAN:
            mod_channel = self.client.mod_channels[self.message.guild.id]
            await mod_channel.send(f'The user {self.message.author.name} has been temporarily banned')
        elif outcome_type == Outcome.PERMANENT_BAN:
            mod_channel = self.client.mod_channels[self.message.guild.id]
            await mod_channel.send(f'The user {self.message.author.name} has been permanently banned')

        self.end_report()