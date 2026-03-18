from serpapi import GoogleSearch

def get_transcript(video_id: str):
    try:
        params = {
            "api_key": '1102bfb02cefcc88b1950f59bb4d8b0e16c6cd3667f3b427d42563497a9f663c',
            'engine': 'youtube_video_transcript',
            'v': video_id
        }

        search = GoogleSearch(params)
        result = search.get_dict()

        return '\n\n'.join(s.get('snippet', '') for s in result['transcript'])
    
    except Exception as e:
        print(f"Error has occured. ", e)
        return ''
   