from enum import Enum, auto
from report import ReportType
import discord
import re


class Outcome(Enum):
    VIOLATE_POLICY = auto()
    VALID_EMERGENCY = auto()
    INVALID_EMERGENCY = auto()


class State(Enum):
    REVIEW_START = auto()
    AWAITING_MESSAGE = auto()
    MESSAGE_IDENTIFIED = auto()
    EMERGENCY_IDENTIFIED = auto()
    REVIEW_COMPLETE = auto()


class Review:
    START_KEYWORD = "review"
    CANCEL_KEYWORD = "cancel"
    HELP_KEYWORD = "help"

    def __init__(self, client):
        self.state = State.REVIEW_START
        self.client = client
        self.message = None
        self.report_info = {}

    async def handle_message(self, message):

        if message.content == self.CANCEL_KEYWORD:
            self.state = State.REVIEW_COMPLETE
            return ["Review cancelled."]

        if self.state == State.REVIEW_START:
            reply = "Thank you for starting the review process. "
            reply += "Say `help` at any time for more information.\n\n"
            reply += "Please copy paste the link to the message you want to review.\n"
            reply += "You can obtain this link by right-clicking the message and clicking `Copy Message Link`."
            self.state = State.AWAITING_MESSAGE

            return [reply]

        if self.state == State.AWAITING_MESSAGE:
            # Parse out the three ID strings from the message link
            m = re.search("/(\d+)/(\d+)/(\d+)", message.content)
            if not m:
                return [
                    "I'm sorry, I couldn't read that link. Please try again or say `cancel` to cancel."
                ]
            guild = self.client.get_guild(int(m.group(1)))
            if not guild:
                return [
                    "I cannot accept reviews of messages from guilds that I'm not in. Please have the guild owner add me to the guild and try again."
                ]
            channel = guild.get_channel(int(m.group(2)))
            if not channel:
                return [
                    "It seems this channel was deleted or never existed. Please try again or say `cancel` to cancel."
                ]
            try:
                message = await channel.fetch_message(int(m.group(3)))
            except discord.errors.NotFound:
                return [
                    "It seems this message was deleted or never existed. Please try again or say `cancel` to cancel."
                ]

            if message not in self.client.pending_reports:
                return [
                    "It seems this message was never reported or has already been reviewed. Please try again or say `cancel` to cancel."
                ]

            self.message = message
            self.report_info = self.client.pending_reports[self.message]
            self.state = State.MESSAGE_IDENTIFIED
            if self.report_info["is_emergency"]:
                self.state = State.EMERGENCY_IDENTIFIED
                reply = (
                    f"Report Description:\n{self.report_info['report_description']}\n"
                )
                reply += "Does this message constitute a serious threat or emergency that requires law enforcement involvement?"
                return [reply]

            if self.report_info["report_category"] == ReportType.OTHER:
                reply = (
                    f"Report Description:\n{self.report_info['report_description']}\n\n"
                )
                reply += "Does this message fall under any of the following categories (Hate Speech, Violence and Gore, Self-Harm/Suicidal Content, Harassment) or violate any of our content policies?\n"
                reply += "Please remember to review the message clearly and carefully!"
                return [reply]
            else:
                reply = (
                    f"Report Description:\n{self.report_info['report_description']}\n"
                )
                reply += f"Does this message fall under the category of {self.report_info['report_sub_category']} or violate another content policy?\n"
                reply += "Please remember to review the message clearly and carefully!"
                return [reply]

        if self.state == State.MESSAGE_IDENTIFIED:
            is_yes = True if message.content == "yes" else False
            if is_yes:
                partial_reply = "Thank you for reviewing this report. The post will be removed and the user may be temporarily or permanently banned"
                reply = await self.handle_outcome(Outcome.VIOLATE_POLICY)
                reply.insert(0, partial_reply)
                return reply
            else:
                self.end_report()
                return [
                    "Thank you for reviewing this report. No additional action will be taken and the report will be closed"
                ]

        if self.state == State.EMERGENCY_IDENTIFIED:
            is_yes = True if message.content == "yes" else False
            if is_yes:
                return await self.handle_outcome(Outcome.VALID_EMERGENCY)
            else:
                return await self.handle_outcome(Outcome.INVALID_EMERGENCY)

        return []

    def end_report(self):
        self.client.pending_reports.pop(self.message)
        self.state = State.REVIEW_COMPLETE
        self.message = None

    def review_complete(self):
        return self.state == State.REVIEW_COMPLETE

    async def handle_outcome(self, outcome_type):
        if outcome_type == Outcome.VIOLATE_POLICY:
            message_author = self.message.author.name
            if message_author in self.client.user_violations:
                self.client.user_violations[message_author] += 1
            else:
                self.client.user_violations[message_author] = 1

            await self.message.author.send(
                "SYSTEM MESSAGE: A violation has been detected resulting in post removal. Additional violations will result in bans."
            )
            await self.message.delete()
            self.end_report()
            if self.client.user_violations[message_author] >= 10:  # 3rd threshold
                return [
                    f"SYSTEM MESSAGE: The user {message_author} has been permanently banned"
                ]
            elif self.client.user_violations[message_author] >= 5:  # 2nd threshold
                return [
                    f"SYSTEM MESSAGE: The user {message_author} has been temporarily banned"
                ]
            elif self.client.user_violations[message_author] >= 2:  # 1st threshold
                return [
                    f"SYSTEM MESSAGE: The user {message_author} has been temporarily shadow banned"
                ]
            else:
                return []

        elif outcome_type == Outcome.VALID_EMERGENCY:
            self.client.pending_reports[self.message]["valid_emergency_count"] += 1
            if self.client.pending_reports[self.message]["valid_emergency_count"] >= 2:
                self.end_report()
                return ["EMERGENCY: Local law enforcement has been contacted!"]
            else:
                valid_emergency_count = self.client.pending_reports[self.message][
                    "valid_emergency_count"
                ]
                self.state = State.REVIEW_COMPLETE
                self.message = None
                if valid_emergency_count == 0:
                    return [
                        f"Thank you for your review! There has been a moderator disagreement so additional reviews are required from moderators."
                    ]
                else:
                    return [
                        f"Thank you for your review! 1 additional review is required from moderators for this report."
                    ]
        elif outcome_type == Outcome.INVALID_EMERGENCY:
            self.client.pending_reports[self.message]["valid_emergency_count"] -= 1
            if self.client.pending_reports[self.message]["valid_emergency_count"] <= -2:
                self.end_report()
                reply = "Thank you for your report. Our moderation team has reviewed the content you flagged and, while we understand your concern, it does not meet our criteria for an emergency situation under our current policies."
                reply += "We appreciate you taking the time to look out for the safety of the community. If you continue to feel that someone is at risk or in danger, and it’s an immediate emergency, we encourage you to contact local authorities directly."
                reply += "If you have any further concerns, feel free to reach out or file another report."
                return [reply]
            else:
                valid_emergency_count = self.client.pending_reports[self.message][
                    "valid_emergency_count"
                ]
                self.state = State.REVIEW_COMPLETE
                self.message = None
                if valid_emergency_count == 0:
                    return [
                        f"Thank you for your review! There has been a moderator disagreement so additional reviews are required from moderators."
                    ]
                else:
                    return [
                        f"Thank you for your review! 1 additional review is required from moderators for this report."
                    ]

        # elif outcome_type == Outcome.TEMP_BAN:
        #     mod_channel = self.client.mod_channels[self.message.guild.id]
        #     await mod_channel.send(
        #         f"The user {self.message.author.name} has been temporarily banned"
        #     )
        # elif outcome_type == Outcome.PERMANENT_BAN:
        #     mod_channel = self.client.mod_channels[self.message.guild.id]
        #     await mod_channel.send(
        #         f"The user {self.message.author.name} has been permanently banned"
        #     )

        # self.end_report()
