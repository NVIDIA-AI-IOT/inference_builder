    async def {{ name }}(self, interface, request, file):
        if not file:
            return 400, "No file provided"
        try:
            asset = self._asset_manager.save_file(file.file, file.filename, "none", file.content_type)
        except Exception as e:
            return 500, str(e)
        response = self.process_response("{{ name }}", request, asdict(asset))
        return 200, response

