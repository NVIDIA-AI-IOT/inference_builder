    async def {{ name }}(self, request):
        try:
            assets = [asset for asset in self._asset_manager.list_assets() if asset.file_name]
        except Exception as e:
            return 500, str(e)
        response = self.process_response("{{ name }}", request, {"assets": assets})
        return 200, response
