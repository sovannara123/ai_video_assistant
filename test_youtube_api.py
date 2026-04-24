# test_youtube_api.py
# paste and run this first!

from youtube_transcript_api import YouTubeTranscriptApi

print("📋 Available methods:")
methods = [
    m for m in dir(YouTubeTranscriptApi)
    if not m.startswith('_')
]
for method in methods:
    print(f"   → {method}")

print("\n🧪 Testing with version 1.2.4...")

TEST_VIDEO_ID = "aircAruvnKk"

try:
    # Version 1.2.x new way
    api    = YouTubeTranscriptApi()
    result = api.fetch(TEST_VIDEO_ID)

    print(f"\n✅ Success!")
    print(f"Type     : {type(result)}")
    print(f"Dir      : {[x for x in dir(result) if not x.startswith('_')]}")

    # Try to iterate
    items = list(result)
    print(f"Items    : {len(items)}")
    print(f"First    : {items[0]}")
    print(f"FirstType: {type(items[0])}")
    print(f"FirstDir : {[x for x in dir(items[0]) if not x.startswith('_')]}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

    