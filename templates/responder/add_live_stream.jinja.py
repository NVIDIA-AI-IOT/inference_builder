    async def {{ name }}(self, request, body, **kwargs):
        stream_info = self.process_request("{{ name }}", body)
        if "url" not in stream_info or not stream_info["url"]:
            return 400, "No url provided"
        try:
            asset = self._asset_manager.add_live_stream(
                stream_info["url"],
                stream_info["description"],
                stream_info["username"],
                stream_info["password"])
        except Exception as e:
            return 500, str(e)
        response = self.process_response("{{ name }}", request, asdict(asset))
        return 200, response

