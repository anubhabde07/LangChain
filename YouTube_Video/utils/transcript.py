from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def get_transcript(video_id: str) -> str:
    try:
        transcript_obj = YouTubeTranscriptApi()
        fetched_transcript = transcript_obj.fetch(video_id=video_id, languages=['en'])
        transcript_list = [snippet.text for snippet in fetched_transcript.snippets]
        return " ".join(transcript_list)
    except TranscriptsDisabled:
        return ""