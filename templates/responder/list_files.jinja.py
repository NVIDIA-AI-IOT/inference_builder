    async def {{ name }}(self, interface, request):
        try:
            assets = self._asset_manager.list_assets()
        except Exception as e:
            return 500, str(e)
        response = self.process_response("{{ name }}", request, {"assets": assets})
        return 200, response
