from enum import Enum, auto
import discord
import re


class State(Enum):
    REPORT_START = auto()  # user messaged bot with `report`
    AWAITING_MESSAGE = auto()  # waiting for user to provide link to message
    MESSAGE_IDENTIFIED = auto()  # message correctly identified
    PROCESSING_REPORT = auto()  # prompt user to categorize report type
    PROCESSING_RESPONSE = auto()  # handle response to report type
    REPORT_COMPLETE = auto()


class ReportType(Enum):
    HATE_SPEECH = "Hate Speech"
    VIOLENCE = "Violence"
    SELF_HARM = "Self-Harm/Suicide"
    HARASSMENT = "Harassment"
    OTHER = "Other"


class Response(Enum):
    IS_ONGOING_PATTERN = auto()
    IS_EMERGENCY = auto()
    BLOCK_USER = auto()


class Report:
    START_KEYWORD = "report"
    CANCEL_KEYWORD = "cancel"
    HELP_KEYWORD = "help"

    def __init__(self, client):
        self.state = State.REPORT_START
        self.client = client
        self.message = None
        self.needs_review = False
        self.is_emergency = False
        self.report_type = None
        self.report_sub_type = None
        self.report_description = ""
        self.response_state = None

    async def handle_message(self, message):
        """
        This function makes up the meat of the user-side reporting flow. It defines how we transition between states and what
        prompts to offer at each of those states. You're welcome to change anything you want; this skeleton is just here to
        get you started and give you a model for working with Discord.
        """

        if message.content == self.CANCEL_KEYWORD:
            self.end_report()
            return ["Report cancelled."]

        if self.state == State.REPORT_START:
            reply = "Thank you for starting the reporting process. "
            reply += "Say `help` at any time for more information.\n\n"
            reply += "Please copy paste the link to the message you want to report.\n"
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
                    "I cannot accept reports of messages from guilds that I'm not in. Please have the guild owner add me to the guild and try again."
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

            self.message = message
            # Here we've found the message - it's up to you to decide what to do next!
            self.state = State.MESSAGE_IDENTIFIED
            # return ["I found this message:", "```" + message.author.name + ": " + message.content + "```", \
            #         "This is all I know how to do right now - it's up to you to build out the rest of my reporting flow!"]

            reply = "What would you like to report? These are some options: \n\n"
            reply += "1) Hate Speech\n"
            reply += "2) Violence and Gore\n"
            reply += "3) Self Harm / Suicidal Content\n"
            reply += "4) Harassment\n"
            reply += "5) Other \n\n"
            reply += "Please enter the number corresponding with the violation you would like to report."

            return [reply]

        if self.state == State.MESSAGE_IDENTIFIED:
            error_message = ["It seems that your input was invalid. Please try again."]
            try:
                report_num = int(message.content)
                self.state = State.PROCESSING_REPORT
                if report_num == 1:
                    self.report_type = ReportType.HATE_SPEECH
                    reply = "What sort of hate speech do you believe is contained in the message?\n\n"
                    reply += "1) Homophobia\n"
                    reply += "2) Racism\n"
                    reply += "3) Sexism\n"
                    reply += "4) Antisemitism\n"
                    reply += "5) Islamophobia"
                    return [reply]
                elif report_num == 2:
                    self.report_type = ReportType.VIOLENCE
                    reply = "Please select the type of violent content\n\n"
                    reply += "1) Explicit Violent Imagery\n"
                    reply += "2) Animal Cruelty\n"
                    reply += "3) Glorification of Violence\n"
                    reply += "4) Violent Instructions\n"
                    reply += "5) Threats of Violence"
                    return [reply]
                elif report_num == 3:
                    self.report_type = ReportType.SELF_HARM
                    reply = "Please select the type of self-harm content\n\n"
                    reply += "1) Self-Harm Imagery\n"
                    reply += "2) Self-Harm Methods/Instructions\n"
                    reply += "3) Glorification of Self-Harm\n"
                    reply += "4) Active Suicidal Threats\n"
                    reply += "5) Encouraging Self-Harm in Others"
                    return [reply]
                elif report_num == 4:
                    self.report_type = ReportType.HARASSMENT
                    reply = "Please select the type of harassment\n\n"
                    reply += "1) Bullying\n"
                    reply += "2) Sharing Private Information\n"
                    reply += "3) Targeted Hate Speech\n"
                    reply += "4) Sexual Harassment"
                    return [reply]
                elif report_num == 5:
                    self.report_type = ReportType.OTHER
                    reply = "Please explain the problem."
                    return [reply]
                else:
                    return error_message
            except ValueError:
                return error_message

        if self.state == State.PROCESSING_REPORT:
            error_message = ["It seems that your input was invalid. Please try again."]
            if self.report_type == ReportType.HATE_SPEECH:
                try:
                    category_idx = int(message.content) - 1
                    hate_speech_cats = [
                        "Homophobia",
                        "Racism",
                        "Sexism",
                        "Antisemitism",
                        "Islamophobia",
                    ]
                    if category_idx > len(hate_speech_cats):
                        return error_message
                    self.report_description += f"The user has reported Hate Speech in the form of {hate_speech_cats[category_idx]}\n"
                    self.report_sub_type = hate_speech_cats[category_idx]
                    self.state = State.PROCESSING_RESPONSE
                    self.response_state = Response.IS_ONGOING_PATTERN
                    return self.prompt_response()
                except ValueError:
                    return error_message
            elif self.report_type == ReportType.VIOLENCE:
                try:
                    category_idx = int(message.content) - 1
                    violence_cats = [
                        "Explicit Violent Imagery",
                        "Animal Cruelty",
                        "Glorification of Violence",
                        "Violent Instructions",
                        "Threats of Violence",
                    ]
                    if category_idx > len(violence_cats):
                        return error_message
                    self.report_description += f"The user has reported Violent Content in the form of {violence_cats[category_idx]}\n"
                    self.report_sub_type = violence_cats[category_idx]
                    self.state = State.PROCESSING_RESPONSE
                    if category_idx == 4:
                        self.response_state = Response.IS_EMERGENCY
                    else:
                        self.response_state = Response.IS_ONGOING_PATTERN
                    return self.prompt_response()
                except ValueError:
                    return error_message
            elif self.report_type == ReportType.SELF_HARM:
                try:
                    category_idx = int(message.content) - 1
                    self_harm_cats = [
                        "Self-Harm Imagery",
                        "Self-Harm Methods/Instructions",
                        "Glorification of Self-Harm",
                        "Active Suicidal Threats",
                        "Encouraging Self-Harm in Others",
                    ]
                    if category_idx > len(self_harm_cats):
                        return error_message

                    self.report_description += f"The user has reported Self-Harm/Suicide related content in the form of {self_harm_cats[category_idx]}\n"
                    self.report_sub_type = self_harm_cats[category_idx]
                    self.state = State.PROCESSING_RESPONSE
                    if category_idx == 3 or category_idx == 4:
                        self.response_state = Response.IS_EMERGENCY
                    else:
                        self.response_state = Response.IS_ONGOING_PATTERN
                    return self.prompt_response()
                except ValueError:
                    return error_message
            elif self.report_type == ReportType.HARASSMENT:
                try:
                    category_idx = int(message.content) - 1
                    harassment_cats = [
                        "Bullying",
                        "Sharing Private Information",
                        "Targeted Hate Speech",
                        "Sexual Harrassment",
                    ]
                    if category_idx > len(harassment_cats):
                        return error_message

                    self.report_description += f"The user has reported Harassment in the form of {harassment_cats[category_idx]}\n"
                    self.report_sub_type = harassment_cats[category_idx]
                    self.state = State.PROCESSING_RESPONSE
                    self.response_state = Response.IS_ONGOING_PATTERN
                    return self.prompt_response()
                except ValueError:
                    return error_message
            elif self.report_type == ReportType.OTHER:
                self.report_description = message.content
                self.needs_review = True
                self.end_report()
                return self.end_message_no_block()
        if self.state == State.PROCESSING_RESPONSE:
            is_yes = True if message.content == "yes" else False
            if self.response_state == Response.IS_ONGOING_PATTERN:
                if is_yes:
                    self.report_description += (
                        "The user has reported this as an ongoing issue\n"
                    )
                self.response_state = Response.BLOCK_USER
                return self.prompt_response()
            elif self.response_state == Response.IS_EMERGENCY:
                if is_yes:
                    self.needs_review = True
                    self.is_emergency = True
                    self.end_report()
                    reply = "We will have our content moderation team review the message and if appropriate will contact local law enforcement to conduct a wellness check. Thank you for looking out for your friends."
                    return [reply]
                else:
                    self.response_state = Response.BLOCK_USER
                    partial_reply = "We will direct this message to our content moderation team to look into this incident. Thank you for keeping our digital community safe."
                    reply = self.prompt_response()
                    reply.insert(0, partial_reply)
                    return reply
            elif self.response_state == Response.BLOCK_USER:
                if is_yes:
                    self.needs_review = True
                    self.end_report()
                    return self.end_message_block()
                else:
                    self.needs_review = True
                    self.end_report()
                    return self.end_message_no_block()

        return []

    def end_message_no_block(self):
        return [
            "Thank you for reporting. Our content moderation team will review the message and decide on appropriate action. This may include post and/or account removal."
        ]

    def end_message_block(self):
        return [
            f"Thank you for filing a report. {self.message.author.name} has been successsfully blocked. You will no longer see content posted by them. Our content moderation team will review the message and account, and decide on appropriate action. This may include post and/or account removal."
        ]

    def prompt_response(self):
        if self.response_state == Response.IS_ONGOING_PATTERN:
            reply = "Is this an ongoing issue or pattern of behavior?\n"
            reply += "Respond yes or no."
            return [reply]
        elif self.response_state == Response.IS_EMERGENCY:
            reply = "Is this an emergency requiring immediate attention?\n"
            reply += "Respond yes or no."
            return [reply]
        elif self.response_state == Response.BLOCK_USER:
            reply = "Would you like to block this user to prevent them from your feed in the future?\n"
            reply += "Respond yes or no."
            return [reply]

    def end_report(self):
        self.state = State.REPORT_COMPLETE
        if self.needs_review:
            self.client.pending_reports[self.message] = {
                "report_category": self.report_type,
                "report_sub_category": self.report_sub_type,
                "report_description": self.report_description,
                "is_emergency": self.is_emergency,
                "valid_emergency_count": 0,
            }

    def report_complete(self):
        return self.state == State.REPORT_COMPLETE
